"""
VN1_train2.py – Stage-2 self-play training for ValueNet1

Full 3-ply minimax (no alpha-beta) with batched ValueNet leaf evaluation.
Root, depth-1, and depth-2 positions become training examples whose targets
are the negamax-backed-up values.  Training data is persisted in a rolling
window on disk so old, stale generations are automatically pruned.
"""

import argparse
import csv
import math
import random
import time
from pathlib import Path
import torch.multiprocessing as mp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from onitama import Boardstate
from onitama.opponents import ab_search_bot
from onitama.opponents.mcts_bot import MCTSBot
from onitama.opponents.random_bot import RandomBot
from onitama.opponents.NN.architecture.ValueNet1 import ValueNet1
from onitama.opponents.NN.VN1_train import (
    encode_state_for_value_net,
    build_card_flip_map,
    augment_with_horizontal_flip,
    update_ema,
)

# C++ engine (optional — dramatically speeds up tree expansion)
try:
    from onitama.engine.onitama_engine import (
        BoardState as CppBoardState,
        MinimaxTree as CppMinimaxTree,
        ABSearchBot as CppABSearchBot,
        MCTSBot as CppMCTSBot,
        WinColour as CppWinColour,
    )
    import numpy as np
    _CPP_ENGINE = True
except ImportError:
    _CPP_ENGINE = False


# ------------------------------------------------------------------ #
#  Search tree (full minimax, no pruning)                             #
# ------------------------------------------------------------------ #

def build_search_tree(board, depth, max_depth):
    """Recursively expand every legal move down to *max_depth* plies."""
    board_tensor, cards_tensor = encode_state_for_value_net(board)

    node = {
        "bt": board_tensor,
        "ct": cards_tensor,
        "children": [],          # list of (move, child_node)
        "terminal": False,
        "value": None,
        "depth": depth,
    }

    result = board.is_won()
    if result is not None:
        winner = result[0]
        current = board.turn_colour()
        node["terminal"] = True
        node["value"] = 1.0 if winner == current else -1.0
        return node

    if depth >= max_depth:
        return node                  # leaf – batch-evaluated later

    for move in board.possible_moves():
        undo = board.apply_move(move)
        child = build_search_tree(board, depth + 1, max_depth)
        node["children"].append((move, child))
        board.undo_move(undo)

    return node


def collect_leaves(node, leaves):
    """Gather non-terminal leaf nodes that need a ValueNet forward pass."""
    if node["terminal"]:
        return
    if not node["children"]:
        leaves.append(node)
        return
    for _, child in node["children"]:
        collect_leaves(child, leaves)


def batch_evaluate_leaves(leaves, model, device):
    """Single batched forward pass over all leaf nodes."""
    if not leaves:
        return
    boards = torch.stack([n["bt"] for n in leaves])
    cards = torch.stack([n["ct"] for n in leaves])
    boards = boards.to(device, non_blocking=True)
    cards = cards.to(device, non_blocking=True)

    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        vals = model(boards, cards).squeeze(1)

    vals = vals.float().cpu()
    for i, node in enumerate(leaves):
        node["value"] = vals[i].item()


def backup_values(node):
    """Negamax back-up: value = max_child(-child_value)."""
    if node["terminal"] or not node["children"]:
        return                       # already has a value
    for _, child in node["children"]:
        backup_values(child)
    node["value"] = max(-child["value"] for _, child in node["children"])


def collect_examples(node, examples, max_depth):
    """Collect (board_tensor, cards_tensor, value) for every internal node."""
    if node["value"] is None:
        return
    if node["depth"] < max_depth:
        examples.append((node["bt"], node["ct"], node["value"]))
    for _, child in node["children"]:
        collect_examples(child, examples, max_depth)


def best_move_from_tree(tree):
    """Return the root's best move (highest negamax value)."""
    if not tree["children"]:
        return None
    return max(tree["children"], key=lambda mc: -mc[1]["value"])[0]


def select_move_with_temperature(tree, temperature):
    """Softmax move selection over negamax child values for exploration.

    The training *targets* are still the deterministic negamax values from the
    tree – temperature only affects which game trajectory gets explored,
    exposing the model to a wider variety of in-game positions.
    """
    if not tree["children"]:
        return None
    if len(tree["children"]) == 1:
        return tree["children"][0][0]

    # Value of each child from the root player's perspective
    values = [-child["value"] for _, child in tree["children"]]

    # Numerically stable softmax
    max_v = max(values)
    exp_values = [math.exp((v - max_v) / temperature) for v in values]
    total = sum(exp_values)
    probs = [e / total for e in exp_values]

    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return tree["children"][i][0]
    return tree["children"][-1][0]  # float rounding fallback


# ------------------------------------------------------------------ #
#  Tree-search gauntlet                                               #
# ------------------------------------------------------------------ #

def _play_gauntlet_game_tree(model, device, search_depth, heuristic_bot, max_plies):
    """Play one gauntlet game; ValueNet plays RED, heuristic plays BLUE."""
    board = Boardstate.Boardstate()
    for _ in range(max_plies):
        terminal = board.is_won()
        if terminal:
            return terminal[0]
        if board.turn_colour() == "RED":
            tree = build_search_tree(board, 0, search_depth)
            leaves = []
            collect_leaves(tree, leaves)
            batch_evaluate_leaves(leaves, model, device)
            backup_values(tree)
            move = best_move_from_tree(tree)
        else:
            move = heuristic_bot.request_move(board)
        if move is None:
            break
        board.apply_move(move)
    terminal = board.is_won()
    if terminal:
        return terminal[0]
    return None


def _gauntlet_worker(args_tuple):
    """Worker for parallel gauntlet. Runs entirely on CPU."""
    state_dict, search_depth, heuristic_depth, max_plies, seed = args_tuple
    random.seed(seed)
    torch.manual_seed(seed)

    model = ValueNet1()
    model.load_state_dict(state_dict)
    model.eval()

    if _CPP_ENGINE:
        cpp_board = CppBoardState.random_start(seed)
        heuristic_bot = CppABSearchBot(heuristic_depth)
        for _ in range(max_plies):
            if cpp_board.is_terminal():
                break
            if cpp_board.is_red_turn():
                tree = CppMinimaxTree.build(cpp_board, search_depth)
                _cpp_evaluate_tree_leaves(tree, model)
                move = tree.best_move()
            else:
                move = heuristic_bot.request_move(cpp_board, -1.0)
            if not move.valid():
                break
            cpp_board.apply_move(move)
        win = cpp_board.is_won()
        if win and win.colour == CppWinColour.RED:
            return 1.0
        elif not win:
            return 0.5
        return 0.0

    # Fallback: Python implementation
    device = torch.device("cpu")
    heuristic_bot = ab_search_bot.ABSearchBot(max_depth=heuristic_depth)
    result = _play_gauntlet_game_tree(model, device, search_depth, heuristic_bot, max_plies)

    if result == "RED":
        return 1.0
    elif result is None:
        return 0.5
    return 0.0


def run_gauntlet_tree_search(
    model, train_device, gauntlet_device,
    heuristic_depth, search_depth, max_plies, num_games,
    num_workers=None, progress_desc=None,
):
    """Gauntlet using batched minimax tree-search for the ValueNet side.

    ValueNet always plays RED; Boardstate.__init__ handles starting-player
    randomisation so colour symmetry is covered across games.
    Games run in parallel across CPU processes.
    """
    if num_workers is None:
        num_workers = min(num_games, mp.cpu_count())
    num_workers = min(num_workers, num_games)

    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    worker_args = [
        (state_dict, search_depth, heuristic_depth, max_plies, i)
        for i in range(num_games)
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_gauntlet_worker, worker_args),
            total=num_games,
            desc=progress_desc or "Gauntlet",
            leave=False,
        ))

    return sum(results) / num_games


# ------------------------------------------------------------------ #
#  Self-play                                                          #
# ------------------------------------------------------------------ #

def self_play_game(model, device, search_depth, max_plies):
    """Play one full game of self-play; return training examples."""
    board = Boardstate.Boardstate()
    all_examples = []

    for _ in range(max_plies):
        if board.is_won() is not None:
            break

        tree = build_search_tree(board, 0, search_depth)

        leaves = []
        collect_leaves(tree, leaves)
        batch_evaluate_leaves(leaves, model, device)
        backup_values(tree)

        examples = []
        collect_examples(tree, examples, 2) # only root and depth-1 nodes become training examples
        all_examples.extend(examples)

        move = best_move_from_tree(tree)
        if move is None:
            break
        board.apply_move(move)

    return all_examples


# ------------------------------------------------------------------ #
#  Parallel self-play (process pool, CPU inference)                    #
# ------------------------------------------------------------------ #

def _self_play_worker(args_tuple):
    """Worker for parallel self-play. Runs entirely on CPU."""
    state_dict, search_depth, max_plies, seed = args_tuple
    random.seed(seed)
    torch.manual_seed(seed)

    model = ValueNet1()
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cpu")
    board = Boardstate.Boardstate()
    all_examples = []

    for _ in range(max_plies):
        if board.is_won() is not None:
            break

        tree = build_search_tree(board, 0, search_depth)
        leaves = []
        collect_leaves(tree, leaves)
        batch_evaluate_leaves(leaves, model, device)
        backup_values(tree)

        examples = []
        collect_examples(tree, examples, 2) # only root and depth-1 nodes become training examples
        all_examples.extend(examples)

        move = best_move_from_tree(tree)
        if move is None:
            break
        board.apply_move(move)

    return [(bt.clone(), ct.clone(), v) for bt, ct, v in all_examples]


def parallel_self_play(model, num_games, search_depth, max_plies, seed,
                       num_workers=None):
    """Run self-play games in parallel across CPU processes."""
    if num_workers is None:
        num_workers = min(num_games, mp.cpu_count())
    num_workers = min(num_workers, num_games)

    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    worker_args = [
        (state_dict, search_depth, max_plies, seed + i)
        for i in range(num_games)
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_self_play_worker, worker_args),
            total=num_games,
            desc=f"Self-play ({num_workers} workers)",
            leave=False,
        ))

    all_examples = []
    for game_examples in results:
        all_examples.extend(game_examples)
    return all_examples


# ------------------------------------------------------------------ #
#  Mixed training games (self-play + opponent play)                    #
# ------------------------------------------------------------------ #

def _play_self_play_training_game(model, device, search_depth, max_plies,
                                  temperature):
    """Self-play with softmax exploration noise; return training examples."""
    board = Boardstate.Boardstate()
    all_examples = []

    for _ in range(max_plies):
        if board.is_won() is not None:
            break

        tree = build_search_tree(board, 0, search_depth)
        leaves = []
        collect_leaves(tree, leaves)
        batch_evaluate_leaves(leaves, model, device)
        backup_values(tree)

        examples = []
        collect_examples(tree, examples, 2)
        all_examples.extend(examples)

        if temperature > 0:
            move = select_move_with_temperature(tree, temperature)
        else:
            move = best_move_from_tree(tree)
        if move is None:
            break
        board.apply_move(move)

    return [(bt.clone(), ct.clone(), v) for bt, ct, v in all_examples]


def _play_vs_opponent_training_game(model, device, search_depth, max_plies,
                                    opponent):
    """ValueNet (RED) vs external opponent (BLUE); return examples from
    ValueNet's search trees only.  The opponent provides out-of-distribution
    positions the model would never see in pure self-play."""
    board = Boardstate.Boardstate()
    all_examples = []

    for _ in range(max_plies):
        if board.is_won() is not None:
            break

        if board.turn_colour() == "RED":
            # ValueNet's turn – build tree, collect examples, play best move
            tree = build_search_tree(board, 0, search_depth)
            leaves = []
            collect_leaves(tree, leaves)
            batch_evaluate_leaves(leaves, model, device)
            backup_values(tree)

            examples = []
            collect_examples(tree, examples, 2)
            all_examples.extend(examples)

            move = best_move_from_tree(tree)
        else:
            # Opponent's turn
            move = opponent.request_move(board)

        if move is None:
            break
        board.apply_move(move)

    return [(bt.clone(), ct.clone(), v) for bt, ct, v in all_examples]


# ------------------------------------------------------------------ #
#  C++ engine accelerated training games                              #
# ------------------------------------------------------------------ #

def _cpp_evaluate_tree_leaves(tree, model):
    """Evaluate minimax tree leaves with the NN model (CPU only)."""
    boards_np, cards_np, n_leaves = tree.get_leaf_tensors()
    if n_leaves > 0:
        boards_t = torch.from_numpy(np.ascontiguousarray(boards_np))
        cards_t = torch.from_numpy(np.ascontiguousarray(cards_np))
        with torch.no_grad():
            vals = model(boards_t, cards_t).squeeze(1)
        tree.set_leaf_values(vals.numpy())
    tree.backup()


def _cpp_collect_examples(tree, example_depth=2):
    """Extract training examples from a backed-up minimax tree."""
    ex_boards, ex_cards, ex_values = tree.get_training_examples(example_depth)
    n = ex_values.shape[0]
    if n == 0:
        return []
    boards_t = torch.from_numpy(ex_boards.copy())
    cards_t = torch.from_numpy(ex_cards.copy())
    return [(boards_t[i], cards_t[i], float(ex_values[i])) for i in range(n)]


def _cpp_play_self_play_training_game(model, search_depth, max_plies,
                                      temperature, seed):
    """Self-play using C++ engine for tree expansion."""
    cpp_board = CppBoardState.random_start(seed)
    all_examples = []
    for _ in range(max_plies):
        if cpp_board.is_terminal():
            break
        tree = CppMinimaxTree.build(cpp_board, search_depth)
        _cpp_evaluate_tree_leaves(tree, model)
        all_examples.extend(_cpp_collect_examples(tree))
        if temperature > 0:
            move = tree.select_move_temperature(temperature)
        else:
            move = tree.best_move()
        if not move.valid():
            break
        cpp_board.apply_move(move)
    return all_examples


def _cpp_play_vs_opponent_training_game(model, search_depth, max_plies,
                                        opponent_type, opponent_kwargs, seed):
    """ValueNet (RED) vs C++ opponent (BLUE), using C++ engine."""
    cpp_board = CppBoardState.random_start(seed)

    if opponent_type == "ab":
        opponent = CppABSearchBot(opponent_kwargs.get("depth", 4))
    elif opponent_type == "mcts":
        time_ms = int(opponent_kwargs.get("time_limit", 0.5) * 1000)
        opponent = CppMCTSBot(time_ms=time_ms)
    else:
        opponent = None  # random — pick uniformly

    all_examples = []
    for _ in range(max_plies):
        if cpp_board.is_terminal():
            break
        if cpp_board.is_red_turn():
            # ValueNet's turn — build tree, collect examples
            tree = CppMinimaxTree.build(cpp_board, search_depth)
            _cpp_evaluate_tree_leaves(tree, model)
            all_examples.extend(_cpp_collect_examples(tree))
            move = tree.best_move()
        else:
            # Opponent's turn
            if opponent is not None:
                if opponent_type == "ab":
                    move = opponent.request_move(cpp_board, -1.0)
                else:
                    move = opponent.request_move(cpp_board)
            else:
                moves = cpp_board.possible_moves()
                move = random.choice(moves) if moves else None
        if move is None or (hasattr(move, 'valid') and not move.valid()):
            break
        cpp_board.apply_move(move)
    return all_examples


def _training_game_worker(args_tuple):
    """Unified worker for all training game types.  Runs entirely on CPU."""
    state_dict, game_type, search_depth, max_plies, seed, kwargs = args_tuple
    random.seed(seed)
    torch.manual_seed(seed)

    model = ValueNet1()
    model.load_state_dict(state_dict)
    model.eval()

    if _CPP_ENGINE:
        if game_type == "self":
            return _cpp_play_self_play_training_game(
                model, search_depth, max_plies,
                temperature=kwargs.get("temperature", 0.15),
                seed=seed,
            )
        else:
            return _cpp_play_vs_opponent_training_game(
                model, search_depth, max_plies,
                opponent_type=game_type,
                opponent_kwargs=kwargs,
                seed=seed,
            )

    # ---------- Fallback: Python implementation ---------- #
    device = torch.device("cpu")

    if game_type == "self":
        return _play_self_play_training_game(
            model, device, search_depth, max_plies,
            temperature=kwargs.get("temperature", 0.15),
        )
    elif game_type == "ab":
        opponent = ab_search_bot.ABSearchBot(
            max_depth=kwargs.get("depth", 4),
        )
    elif game_type == "mcts":
        opponent = MCTSBot(time_limit=kwargs.get("time_limit", 0.5))
    elif game_type == "random":
        opponent = RandomBot()
    else:
        raise ValueError(f"Unknown game type: {game_type}")

    return _play_vs_opponent_training_game(
        model, device, search_depth, max_plies, opponent,
    )


def parallel_training_games(model, game_specs, search_depth, max_plies, seed,
                            num_workers=None):
    """Run a mixed batch of training games in parallel.

    *game_specs* is a list of ``(game_type, kwargs)`` tuples – one per game.
    All game types share a single process pool for maximum CPU utilisation.
    """
    total_games = len(game_specs)
    if total_games == 0:
        return []
    if num_workers is None:
        num_workers = min(total_games, mp.cpu_count())
    num_workers = min(num_workers, total_games)

    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    worker_args = [
        (state_dict, game_type, search_depth, max_plies, seed + i, kwargs)
        for i, (game_type, kwargs) in enumerate(game_specs)
    ]

    # Compact description of the game mix for the progress bar
    type_counts = {}
    for game_type, _ in game_specs:
        type_counts[game_type] = type_counts.get(game_type, 0) + 1
    mix_desc = "+".join(f"{n}{t}" for t, n in type_counts.items())

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_training_game_worker, worker_args),
            total=total_games,
            desc=f"Training games [{mix_desc}] ({num_workers}w)",
            leave=False,
        ))

    all_examples = []
    for game_examples in results:
        all_examples.extend(game_examples)
    return all_examples


# ------------------------------------------------------------------ #
#  Rolling-window persistence                                         #
# ------------------------------------------------------------------ #

def save_generation(data_dir, examples, generation):
    """Save one generation of training examples to disk."""
    data_dir.mkdir(parents=True, exist_ok=True)
    boards = torch.stack([e[0] for e in examples])
    cards = torch.stack([e[1] for e in examples])
    values = torch.tensor([e[2] for e in examples], dtype=torch.float32).unsqueeze(1)
    path = data_dir / f"gen_{generation:04d}.pt"
    torch.save({"boards": boards, "cards": cards, "values": values}, path)
    return path


def load_rolling_window(data_dir, window_size):
    """Load the last *window_size* generation files; delete older ones."""
    if not data_dir.exists():
        return None, None, None

    files = sorted(data_dir.glob("gen_*.pt"))
    if not files:
        return None, None, None

    if len(files) > window_size:
        for old in files[:-window_size]:
            old.unlink()
        files = files[-window_size:]

    boards_l, cards_l, values_l = [], [], []
    for f in files:
        p = torch.load(f, map_location="cpu")
        boards_l.append(p["boards"])
        cards_l.append(p["cards"])
        values_l.append(p["values"])

    return torch.cat(boards_l), torch.cat(cards_l), torch.cat(values_l)


# ------------------------------------------------------------------ #
#  Training step                                                      #
# ------------------------------------------------------------------ #

def train_epoch(
    model, optimizer, criterion, boards, cards, values,
    batch_size, device, progress_desc=None,
    num_workers=0, pin_memory=False, prefetch_factor=2,
    persistent_workers=False,
):
    dataset = TensorDataset(boards, cards, values)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    model.train()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    total_loss, total_n = 0.0, 0
    it = tqdm(loader, desc=progress_desc, leave=False) if progress_desc else loader

    for b_board, b_cards, b_target in it:
        b_board = b_board.to(device, non_blocking=True)
        b_cards = b_cards.to(device, non_blocking=True)
        b_target = b_target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred = model(b_board, b_cards)
            loss = criterion(pred, b_target)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()

        n = b_board.size(0)
        total_loss += loss.item() * n
        total_n += n

    return total_loss / max(total_n, 1)


# ------------------------------------------------------------------ #
#  Logging & plotting                                                 #
# ------------------------------------------------------------------ #

CSV_COLUMNS = ["iteration", "positions", "avg_loss", "gauntlet_win_rate", "ema_win_rate"]


def init_csv_log(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(CSV_COLUMNS)


def append_csv_log(path, iteration, positions, avg_loss, gauntlet_wr, ema_wr):
    with path.open("a", newline="") as f:
        csv.writer(f).writerow([
            iteration,
            positions,
            f"{avg_loss:.8f}",
            "" if gauntlet_wr is None else f"{gauntlet_wr:.6f}",
            "" if ema_wr is None else f"{ema_wr:.6f}",
        ])


def read_best_ema_from_log(path):
    """Return the highest ema_win_rate recorded in the CSV log, or 0.0."""
    if not path.exists():
        return 0.0
    best = 0.0
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("ema_win_rate", "")
            if val:
                try:
                    best = max(best, float(val))
                except ValueError:
                    pass
    return best


def save_training_plot(iterations, losses, ema_wrs, path, log_scale=False):
    if not iterations:
        return
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(iterations, losses, "o-", color="tab:blue", markersize=2, label="Loss")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    valid = [(i, w) for i, w in zip(iterations, ema_wrs) if w is not None]
    if valid:
        ax2.plot(
            [v[0] for v in valid], [v[1] for v in valid],
            "x-", color="tab:green", markersize=3, label="EMA Win Rate",
        )
    ax2.set_ylabel("EMA Win Rate", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    if log_scale:
        ax1.set_yscale("log")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Checkpoint                                                         #
# ------------------------------------------------------------------ #

def save_checkpoint(path, model, optimizer, iteration, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "args": vars(args),
    }, path)


# ------------------------------------------------------------------ #
#  Window regeneration                                                #
# ------------------------------------------------------------------ #

def regenerate_window(model, data_dir, window_size, search_depth, max_plies,
                      games_per_generation, seed, num_workers=None):
    """Delete old window data and generate a fresh window from the current model.

    Runs *window_size* generations of self-play (no training, no gauntlet)
    so the rolling window is fully populated with data from the current
    checkpoint's policy.
    """
    # -- warn and delete old data -- #
    existing = sorted(data_dir.glob("gen_*.pt")) if data_dir.exists() else []
    if existing:
        print(f"\n⚠  --regenerate-window will DELETE {len(existing)} generation "
              f"files in {data_dir}")
        for t in range(3, 0, -1):
            print(f"   Starting in {t}...", flush=True)
            time.sleep(1)
        for f in existing:
            f.unlink()
        print(f"   Deleted {len(existing)} files.")

    data_dir.mkdir(parents=True, exist_ok=True)

    # -- generate fresh window -- #
    print(f"Regenerating window: {window_size} generations, "
          f"{games_per_generation} games each\n")

    for gen_idx in range(window_size):
        gen_seed = seed - window_size + gen_idx
        examples = parallel_self_play(
            model,
            num_games=games_per_generation,
            search_depth=search_depth,
            max_plies=max_plies,
            seed=gen_seed,
            num_workers=num_workers,
        )
        if examples:
            save_generation(data_dir, examples, gen_idx + 1)
        print(f"  Window generation {gen_idx + 1}/{window_size}: "
              f"{len(examples)} examples")

    print(f"Window regeneration complete.\n")


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="VN1 Stage-2: self-play training with batched 3-ply minimax",
    )

    # Self-play / training games
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--self-games-per-iter", type=int, default=24,
                        help="Self-play games per iteration (with exploration noise)")
    parser.add_argument("--ab-search-games-per-iter", type=int, default=8,
                        help="Games vs AB search bot per iteration")
    parser.add_argument("--mcts-games-per-iter", type=int, default=0,
                        help="Games vs MCTS bot per iteration")
    parser.add_argument("--random-games-per-iter", type=int, default=0,
                        help="Games vs random bot per iteration")
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--self-play-temperature", type=float, default=0.15,
                        help="Softmax temperature for self-play move selection (0 = deterministic)")
    parser.add_argument("--ab-opponent-depth", type=int, default=4,
                        help="Search depth for AB opponent in training games")
    parser.add_argument("--mcts-time-limit", type=float, default=0.5,
                        help="Time limit (seconds) for MCTS opponent in training games")

    # Training
    parser.add_argument("--epochs-per-iteration", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true", default=True)

    # Self-play parallelism
    parser.add_argument("--self-play-workers", type=int, default=None,
                        help="Number of parallel self-play processes (default: cpu_count)")

    # Rolling window
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--data-dir", type=str, default="logs2/generations")
    parser.add_argument("--no-flip", action="store_true")

    # Gauntlet
    parser.add_argument("--gauntlet-every", type=int, default=5)
    parser.add_argument("--gauntlet-games", type=int, default=16,
                        help="Number of games per gauntlet evaluation")
    parser.add_argument("--gauntlet-heuristic-depth", type=int, default=4)
    parser.add_argument("--gauntlet-valuenet-depth", type=int, default=None,
                        help="Search depth for ValueNet side of gauntlet (default: --search-depth)")
    parser.add_argument("--gauntlet-plies", type=int, default=160)
    parser.add_argument("--gauntlet-device", type=str, default="cpu",
                        choices=["cuda", "cpu"])

    # I/O
    parser.add_argument("--save-path", type=str, default="logs2/vn1_checkpoint.pt")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Bootstrap model weights from a Stage-1 checkpoint")
    parser.add_argument("--plot-path", type=str, default="logs2/vn1_loss_plot.png")
    parser.add_argument("--plot-every", type=int, default=1)
    parser.add_argument("--log-scale", action="store_true", default=False,
                        help="Use logarithmic scale for loss in the training plot")
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--regenerate-window", action="store_true", default=False,
                        help="Delete existing window data and regenerate from current checkpoint before training")

    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = ValueNet1().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    start_iteration = 0

    # Resume a previous Stage-2 run
    if args.resume and save_path.exists():
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iteration = ckpt.get("iteration", 0)
        print(f"Resumed from {save_path} at iteration {start_iteration}")
    # Or bootstrap from a Stage-1 checkpoint
    elif args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded weights from {args.load_checkpoint}")

    data_dir = Path(args.data_dir)
    flip_map = build_card_flip_map()
    gauntlet_device = torch.device(
        args.gauntlet_device
        if torch.cuda.is_available() or args.gauntlet_device == "cpu"
        else "cpu"
    )

    # ---- regenerate window if requested ---- #
    if args.regenerate_window:
        regenerate_window(
            model=model,
            data_dir=data_dir,
            window_size=args.window_size,
            search_depth=args.search_depth,
            max_plies=args.max_plies,
            games_per_generation=args.self_games_per_iter,
            seed=args.seed + start_iteration * args.self_games_per_iter,
            num_workers=args.self_play_workers,
        )

    default_log = Path(__file__).resolve().parent / "logs2" / "vn1_train2_log.csv"
    log_path = Path(args.log_path) if args.log_path else default_log
    init_csv_log(log_path)

    plot_path = Path(args.plot_path)
    half_life = 5.0

    pb_save_path = Path(args.save_path).parent / "vn1_pb_checkpoint.pt"
    best_ema_wr = read_best_ema_from_log(log_path)
    if best_ema_wr > 0.0:
        print(f"Best EMA win rate from log: {best_ema_wr:.4f}")

    gauntlet_search_depth = (
        args.gauntlet_valuenet_depth
        if args.gauntlet_valuenet_depth is not None
        else args.search_depth
    )

    iter_hist, loss_hist, wr_hist = [], [], []
    ema_wr = None

    for iteration in range(start_iteration, args.iterations):
        random.seed(args.seed + iteration)

        # ---- training games (self-play + opponent play) ---- #
        game_specs = []
        for _ in range(args.self_games_per_iter):
            game_specs.append(("self", {"temperature": args.self_play_temperature}))
        for _ in range(args.ab_search_games_per_iter):
            game_specs.append(("ab", {"depth": args.ab_opponent_depth}))
        for _ in range(args.mcts_games_per_iter):
            game_specs.append(("mcts", {"time_limit": args.mcts_time_limit}))
        for _ in range(args.random_games_per_iter):
            game_specs.append(("random", {}))

        all_examples = parallel_training_games(
            model,
            game_specs=game_specs,
            search_depth=args.search_depth,
            max_plies=args.max_plies,
            seed=args.seed + iteration * len(game_specs),
            num_workers=args.self_play_workers,
        )

        if not all_examples:
            print(f"Iteration {iteration + 1}: no examples generated; skipping")
            continue

        save_generation(data_dir, all_examples, iteration + 1)

        # ---- load rolling window ---- #
        boards, cards, values = load_rolling_window(data_dir, args.window_size)
        if boards is None:
            continue

        if not args.no_flip:
            boards, cards, values = augment_with_horizontal_flip(
                boards, cards, values, flip_map,
            )

        # ---- train ---- #
        losses = []
        for epoch in range(args.epochs_per_iteration):
            desc = f"Iter {iteration + 1} Epoch {epoch + 1}/{args.epochs_per_iteration}"
            loss = train_epoch(
                model, optimizer, criterion, boards, cards, values,
                args.batch_size, device, progress_desc=desc,
                num_workers=args.num_workers, pin_memory=args.pin_memory,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers,
            )
            losses.append(loss)

        avg_loss = sum(losses) / max(len(losses), 1)

        # ---- gauntlet ---- #
        g_wr = None
        if args.gauntlet_every > 0 and (iteration + 1) % args.gauntlet_every == 0:
            g_wr = run_gauntlet_tree_search(
                model=model,
                train_device=device,
                gauntlet_device=gauntlet_device,
                heuristic_depth=args.gauntlet_heuristic_depth,
                search_depth=gauntlet_search_depth,
                max_plies=args.gauntlet_plies,
                num_games=args.gauntlet_games,
                progress_desc=f"Iter {iteration + 1} gauntlet",
            )
            ema_wr = update_ema(ema_wr, g_wr, half_life)
            if ema_wr is not None and ema_wr > best_ema_wr:
                best_ema_wr = ema_wr
                save_checkpoint(pb_save_path, model, optimizer, iteration + 1, args)
                print(f"  New personal best EMA: {best_ema_wr:.4f} → saved to {pb_save_path}")

        iter_hist.append(iteration + 1)
        loss_hist.append(avg_loss)
        wr_hist.append(ema_wr)

        g_disp = "n/a" if g_wr is None else f"{g_wr:.3f}"
        e_disp = "n/a" if ema_wr is None else f"{ema_wr:.3f}"
        print(
            f"Iteration {iteration + 1}/{args.iterations} | "
            f"examples={len(all_examples)} | "
            f"window={boards.size(0)} | "
            f"avg_loss={avg_loss:.6f} | "
            f"gauntlet={g_disp} | ema={e_disp}"
        )

        append_csv_log(log_path, iteration + 1, boards.size(0), avg_loss, g_wr, ema_wr)

        if args.plot_every > 0 and (iteration + 1) % args.plot_every == 0:
            save_training_plot(iter_hist, loss_hist, wr_hist, plot_path, log_scale=args.log_scale)

        save_checkpoint(save_path, model, optimizer, iteration + 1, args)

    print(f"Training complete. Checkpoint saved to {save_path}")


if __name__ == "__main__":
    main()
