"""
VN1_train.py – Train ValueNet from heuristic AB-search self-play

Each iteration plays a batch of games where both sides are controlled by
the C++ heuristic alpha-beta search bot.  At every root position (before
the move is played) the board encoding and the search-backed evaluation
are recorded as a training example.

Training data is persisted in a rolling window on disk so old, stale
generations are automatically pruned.  Gauntlet evaluation pits the
ValueNet-backed AB search against the pure heuristic AB search.

Uses the C++ engine (onitama_engine) when available for dramatically
faster game generation and gauntlet play; falls back to the Python
implementation otherwise.
"""

import argparse
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from onitama import Boardstate
from onitama.Deck import Deck
from onitama.opponents import ab_search_bot
# from onitama.opponents.NN.architecture.ValueNet0 import ValueNet0 as ValueNet
from onitama.opponents.NN.architecture.NHNet import NeuralHeuristicNet as ValueNet

# C++ engine (optional — dramatically speeds up game generation & gauntlet)
try:
    from onitama.engine.onitama_engine import (
        BoardState as CppBoardState,
        ABSearchBot as CppABSearchBot,
        WinColour as CppWinColour,
    )
    import numpy as np
    _CPP_ENGINE = True
except ImportError:
    _CPP_ENGINE = False


# ------------------------------------------------------------------ #
#  Board encoding                                                     #
# ------------------------------------------------------------------ #

def _transform_coord(coord, flip):
    if coord is None:
        return None
    x, y = coord
    if flip:
        return 4 - x, 4 - y
    return x, y


def encode_state_for_value_net(board):
    turn = board.turn_colour()
    is_blue_to_move = turn == "BLUE"

    if is_blue_to_move:
        friendly_master = board.positions[5]
        friendly_students = board.positions[6:10]
        enemy_master = board.positions[0]
        enemy_students = board.positions[1:5]
        player_cards = board.blue_cards
        opp_cards = board.red_cards
        flip = True
    else:
        friendly_master = board.positions[0]
        friendly_students = board.positions[1:5]
        enemy_master = board.positions[5]
        enemy_students = board.positions[6:10]
        player_cards = board.red_cards
        opp_cards = board.blue_cards
        flip = False

    board_tensor = torch.zeros((4, 5, 5), dtype=torch.float32)

    transformed_friendly_master = _transform_coord(friendly_master, flip)
    if transformed_friendly_master is not None:
        x, y = transformed_friendly_master
        board_tensor[0, y, x] = 1.0

    for piece in friendly_students:
        transformed_piece = _transform_coord(piece, flip)
        if transformed_piece is None:
            continue
        x, y = transformed_piece
        board_tensor[1, y, x] = 1.0

    transformed_enemy_master = _transform_coord(enemy_master, flip)
    if transformed_enemy_master is not None:
        x, y = transformed_enemy_master
        board_tensor[2, y, x] = 1.0

    for piece in enemy_students:
        transformed_piece = _transform_coord(piece, flip)
        if transformed_piece is None:
            continue
        x, y = transformed_piece
        board_tensor[3, y, x] = 1.0

    cards_tensor = torch.zeros((3, 16), dtype=torch.float32)
    cards_tensor[0, player_cards[0].idx] = 1.0
    cards_tensor[0, player_cards[1].idx] = 1.0
    cards_tensor[1, opp_cards[0].idx] = 1.0
    cards_tensor[1, opp_cards[1].idx] = 1.0
    cards_tensor[2, board.trans_card.idx] = 1.0

    return board_tensor, cards_tensor


# ------------------------------------------------------------------ #
#  Horizontal flip augmentation                                       #
# ------------------------------------------------------------------ #

def build_card_flip_map():
    moveset_to_idx = {}
    for card in Deck.DECK:
        moveset_key = tuple(sorted(card.moveset))
        moveset_to_idx[moveset_key] = card.idx

    flip_map = {}
    for card in Deck.DECK:
        flipped_moveset = tuple(sorted([(-dx, dy) for dx, dy in card.moveset]))
        flipped_idx = moveset_to_idx.get(flipped_moveset)
        if flipped_idx is None:
            raise ValueError(f"No horizontal flip match for card {card.name}")
        flip_map[card.idx] = flipped_idx
    return flip_map


def flip_cards_tensor(cards_tensor, flip_map):
    flipped = torch.zeros_like(cards_tensor)
    for row in range(cards_tensor.size(0)):
        active_indices = torch.nonzero(cards_tensor[row], as_tuple=False).flatten()
        for idx in active_indices:
            new_idx = flip_map[int(idx.item())]
            flipped[row, new_idx] = cards_tensor[row, idx]
    return flipped


def augment_with_horizontal_flip(boards, cards, values, flip_map):
    flipped_boards = boards.flip(-1)
    flipped_cards = torch.stack(
        [flip_cards_tensor(cards[idx], flip_map) for idx in range(cards.size(0))],
        dim=0,
    )
    augmented_boards = torch.cat([boards, flipped_boards], dim=0)
    augmented_cards = torch.cat([cards, flipped_cards], dim=0)
    augmented_values = torch.cat([values, values.clone()], dim=0)
    return augmented_boards, augmented_cards, augmented_values


# ------------------------------------------------------------------ #
#  Training-data generation (heuristic AB self-play, root positions)  #
# ------------------------------------------------------------------ #

def _cpp_generate_game(ab_depth, max_plies):
    """Play one game using C++ engine; return list of (board_t, cards_t, value)."""
    cpp_board = CppBoardState.random_start(random.randint(0, 2**32 - 1))
    bot = CppABSearchBot(ab_depth)
    examples = []

    for _ in range(max_plies):
        if cpp_board.is_terminal():
            break

        # Encode root position
        b_np, c_np = cpp_board.encode_for_nn()
        board_tensor = torch.from_numpy(np.ascontiguousarray(b_np))
        cards_tensor = torch.from_numpy(np.ascontiguousarray(c_np))

        # AB search returns (eval_red, move)
        eval_red, move = bot.request_move_with_eval(cpp_board)

        # Convert to current-player-relative and normalise to [-1, 1]
        is_red_turn = cpp_board.is_red_turn()
        value_current = eval_red if is_red_turn else -eval_red
        value_normalized = max(-1.0, min(1.0, value_current / 1000.0))

        examples.append((board_tensor, cards_tensor, value_normalized))

        if not move.valid():
            break
        cpp_board.apply_move(move)

    return examples


def _py_generate_game(ab_depth, max_plies):
    """Play one game using Python engine; return list of (board_t, cards_t, value)."""
    bot = ab_search_bot.ABSearchBot(max_depth=ab_depth)
    board = Boardstate.Boardstate()
    examples = []

    for _ in range(max_plies):
        terminal = board.is_won()
        if terminal:
            break

        board_tensor, cards_tensor = encode_state_for_value_net(board)

        value_red, move = bot.recursive_search(
            board, return_move=True, max_depth=ab_depth,
        )

        value_current = value_red if board.turn_colour() == "RED" else -value_red
        value_normalized = max(-1.0, min(1.0, value_current / 1000.0))

        examples.append((board_tensor, cards_tensor, value_normalized))

        if move is None:
            break
        board.apply_move(move)

    return examples


def _generation_worker(args_tuple):
    """Worker for parallel game generation. Runs entirely on CPU."""
    ab_depth, max_plies = args_tuple

    if _CPP_ENGINE:
        return _cpp_generate_game(ab_depth, max_plies)
    return _py_generate_game(ab_depth, max_plies)


def parallel_generate_examples(num_games, ab_depth, max_plies, num_workers=None):
    """Run heuristic AB self-play games in parallel; return example list."""
    if num_workers is None:
        num_workers = min(num_games, mp.cpu_count())
    num_workers = min(num_workers, num_games)

    worker_args = [
        (ab_depth, max_plies) for i in range(num_games)
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_generation_worker, worker_args),
            total=num_games,
            desc=f"AB self-play ({num_workers} workers)",
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
#  Training step                                                     #
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
#  ValueNet leaf evaluator (used by gauntlet's ValueNetABSearchBot)   #
# ------------------------------------------------------------------ #

def evaluate_value_net_red(board, model, device, depth):
    try:
        winner = board.is_won()[0]
        depth_discount = depth * 5
        return 1000 - depth_discount if winner == "RED" else -1000 + depth_discount
    except Exception:
        pass

    board_tensor, cards_tensor = encode_state_for_value_net(board)
    board_tensor = board_tensor.unsqueeze(0).to(device)
    cards_tensor = cards_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        value_current = model(board_tensor, cards_tensor).item()

    value_red = value_current if board.turn_colour() == "RED" else -value_current
    return max(-1000.0, min(1000.0, value_red * 1000.0))


class ValueNetABSearchBot(ab_search_bot.ABSearchBot):
    def __init__(self, model, device, max_depth=2):
        super().__init__(max_depth=max_depth)
        self.model = model
        self.device = device

    def _evaluate_position(self, board, depth=0):
        return evaluate_value_net_red(board, self.model, self.device, depth)


def build_search_tree(board, depth, max_depth):
    """Recursively expand legal moves to max_depth for negamax backup."""
    node = {
        "bt": None,
        "ct": None,
        "children": [],
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
        board_tensor, cards_tensor = encode_state_for_value_net(board)
        node["bt"] = board_tensor
        node["ct"] = cards_tensor
        return node

    for move in board.possible_moves_search_optimised():
        undo = board.apply_move(move)
        child = build_search_tree(board, depth + 1, max_depth)
        node["children"].append((move, child))
        board.undo_move(undo)

    return node


def collect_leaves(node, leaves):
    """Gather non-terminal leaves for one batched ValueNet forward pass."""
    if node["terminal"]:
        return
    if not node["children"]:
        leaves.append(node)
        return
    for _, child in node["children"]:
        collect_leaves(child, leaves)


def batch_evaluate_leaves(leaves, model, device):
    """Evaluate all leaves in one batch on the selected device."""
    if not leaves:
        return

    boards = torch.stack([n["bt"] for n in leaves]).to(device, non_blocking=True)
    cards = torch.stack([n["ct"] for n in leaves]).to(device, non_blocking=True)

    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        vals = model(boards, cards).squeeze(1)

    vals = vals.float().cpu()
    for i, node in enumerate(leaves):
        node["value"] = vals[i].item()


def backup_values(node):
    """Negamax backup: value = max_child(-child_value)."""
    if node["terminal"] or not node["children"]:
        return
    for _, child in node["children"]:
        backup_values(child)
    node["value"] = max(-child["value"] for _, child in node["children"])


def best_move_from_tree(tree):
    if not tree["children"]:
        return None
    return max(tree["children"], key=lambda mc: -mc[1]["value"])[0]


def _play_gauntlet_game_tree_search(model, device, search_depth, heuristic_depth, max_plies):
    """ValueNet plays RED with full tree-search + batched leaf eval."""
    board = Boardstate.Boardstate()

    if _CPP_ENGINE:
        heuristic_bot_cpp = CppABSearchBot(heuristic_depth)
    else:
        heuristic_bot_py = ab_search_bot.ABSearchBot(max_depth=heuristic_depth)

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
            if _CPP_ENGINE:
                cpp_board = CppBoardState.from_python(board)
                cpp_move = heuristic_bot_cpp.request_move(cpp_board, -1.0)
                if not cpp_move.valid():
                    move = None
                else:
                    fx, fy = cpp_move.to_coords()[0]
                    tx, ty = cpp_move.to_coords()[1]
                    card_name = cpp_move.to_coords()[2]
                    move = ((fx, fy), (tx, ty), card_name)
            else:
                move = heuristic_bot_py.request_move(board)

        if move is None:
            break
        board.apply_move(move)

    return None


def run_gauntlet_tree_search_batched(
    model,
    train_device,
    gauntlet_device,
    heuristic_depth,
    search_depth,
    max_plies,
    num_games,
    progress_desc=None,
):
    """Sequential gauntlet where ValueNet uses full-depth tree search.

    Leaf values are evaluated in batches on gauntlet_device (typically GPU).
    """
    original_device = next(model.parameters()).device
    if gauntlet_device != original_device:
        model = model.to(gauntlet_device)

    wins = 0.0
    bar = tqdm(total=num_games, desc=progress_desc or "Gauntlet", leave=False)

    try:
        for _ in range(num_games):
            result = _play_gauntlet_game_tree_search(
                model=model,
                device=gauntlet_device,
                search_depth=search_depth,
                heuristic_depth=heuristic_depth,
                max_plies=max_plies,
            )
            if result == "RED":
                wins += 1.0
            elif result is None:
                wins += 0.5
            bar.update(1)
    finally:
        bar.close()
        if gauntlet_device != original_device:
            model.to(train_device)

    return wins / max(num_games, 1)


# ------------------------------------------------------------------ #
#  Gauntlet (parallel, C++ when available)                            #
# ------------------------------------------------------------------ #

def _gauntlet_worker(args_tuple):
    """Worker for parallel gauntlet. Runs entirely on CPU."""
    state_dict, valuenet_depth, heuristic_depth, max_plies = args_tuple

    model = ValueNet()
    model.load_state_dict(state_dict)
    model.eval()
    device = torch.device("cpu")

    if _CPP_ENGINE:
        cpp_board = CppBoardState.random_start(random.randint(0, 2**32 - 1))
        heuristic_bot = CppABSearchBot(heuristic_depth)
        value_bot = ValueNetABSearchBot(model=model, device=device, max_depth=valuenet_depth)

        # Need a Python Boardstate for the ValueNetABSearchBot
        py_board = cpp_board.to_python()

        for _ in range(max_plies):
            if cpp_board.is_terminal():
                break
            if cpp_board.is_red_turn():
                # ValueNet plays RED via Python AB search with NN leaf eval
                move = value_bot.request_move(py_board)
                if move is None:
                    break
                py_board.apply_move(move)
                # Sync C++ board: convert the Python move to C++ move
                cpp_board = CppBoardState.from_python(py_board)
            else:
                # Heuristic plays BLUE via C++ engine
                move = heuristic_bot.request_move(cpp_board, -1.0)
                if not move.valid():
                    break
                cpp_board.apply_move(move)
                py_board = cpp_board.to_python()

        win = cpp_board.is_won()
        if win and win.colour == CppWinColour.RED:
            return 1.0
        elif not win:
            return 0.5
        return 0.0

    # Fallback: pure Python
    value_bot = ValueNetABSearchBot(model=model, device=device, max_depth=valuenet_depth)
    heuristic_bot = ab_search_bot.ABSearchBot(max_depth=heuristic_depth)

    board = Boardstate.Boardstate()
    for _ in range(max_plies):
        terminal = board.is_won()
        if terminal:
            if terminal[0] == "RED":
                return 1.0
            return 0.0

        bot = value_bot if board.turn_colour() == "RED" else heuristic_bot
        move = bot.request_move(board)
        if move is None:
            break
        board.apply_move(move)

    return 0.5


def run_gauntlet(model, heuristic_depth, valuenet_depth, max_plies,
                 num_games, num_workers=None, progress_desc=None):
    """Gauntlet: ValueNet (RED) vs heuristic AB (BLUE) across many games.

    Games run in parallel across CPU processes.  BoardState.__init__ handles
    starting-player randomisation so colour symmetry is covered across games.
    """
    if num_workers is None:
        num_workers = min(num_games, mp.cpu_count())
    num_workers = min(num_workers, num_games)

    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    worker_args = [
        (state_dict, valuenet_depth, heuristic_depth, max_plies) for _ in range(num_games)
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
#  Utilities: EMA, plotting, checkpoint, CSV logging                  #
# ------------------------------------------------------------------ #

def update_ema(previous, current, half_life):
    decay = 0.5 ** (1.0 / half_life)
    if previous is None:
        return current
    return previous * decay + current * (1.0 - decay)


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


def save_checkpoint(path, model, optimizer, iteration, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "args": vars(args),
    }, path)


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


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Train ValueNet from heuristic AB-search self-play",
    )

    # General
    parser.add_argument("--iterations", type=int, default=1000)

    # I/O
    parser.add_argument("--resume", action="store_false")
    parser.add_argument("--save-file", type=str, default="logs/")
    parser.add_argument("--load-checkpoint", type=str, default=None)
    parser.add_argument("--plot-every", type=int, default=1)
    parser.add_argument("--log-scale", action="store_true", help="Logarithmic loss axes in training loss plot")

    # Data generation
    parser.add_argument("--games-per-iteration", type=int, default=64)
    parser.add_argument("--ab-depth", type=int, default=5)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--generation-workers", type=int, default=None) # Default: CPU count
    parser.add_argument("--no-generate", action="store_true")

    # Training
    parser.add_argument("--epochs-per-iteration", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_false")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_false")

    # Rolling window
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--data-dir", type=str, default=None) # Default: <save_file>/generations/
    parser.add_argument("--no-flip", action="store_true")

    # Gauntlet
    parser.add_argument("--gauntlet-every", type=int, default=5)
    parser.add_argument("--gauntlet-games", type=int, default=16)
    parser.add_argument("--gauntlet-plies", type=int, default=160)
    parser.add_argument("--gauntlet-heuristic-depth", type=int, default=3)
    parser.add_argument("--gauntlet-valuenet-depth", type=int, default=2)
    parser.add_argument("--gauntlet-valuenet-mode",
        type=str,
        default="tree-batched",
        choices=["ab", "tree-batched"],
        help="ab: ValueNet as AB leaf evaluator; tree-batched: full search with batched leaf eval",
    )
    parser.add_argument("--gauntlet-device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device used for gauntlet ValueNet inference (for tree-batched mode)",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"C++ engine: {'available' if _CPP_ENGINE else 'UNAVAILABLE (falling back to Python)'}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = ValueNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    save_file = Path(args.save_file)
    save_file.mkdir(parents=True, exist_ok=True)
    start_iteration = 0

    checkpoint_path = save_file / "vn_checkpoint.pt"

    # Load checkpoint if specified
    if args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded weights from {args.load_checkpoint}")
    # Resume a previous run
    elif args.resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iteration = ckpt.get("iteration", 0)
        print(f"Resumed from {checkpoint_path} at iteration {start_iteration}")
    else:
        print("No checkpoint found. Using default weight initialisation.")

    data_dir = save_file / "generations" if args.data_dir is None else Path(args.data_dir)
    flip_map = build_card_flip_map()

    log_path = save_file / "vn_train_log.csv"
    init_csv_log(log_path)

    plot_path = save_file / "vn_train_plot.png"
    half_life = 2.0
    gauntlet_device = torch.device(
        args.gauntlet_device
        if torch.cuda.is_available() or args.gauntlet_device == "cpu"
        else "cpu"
    )

    save_path = save_file / "vn_checkpoint.pt"
    pb_save_path = save_file / "vn_pb_checkpoint.pt"
    best_ema_wr = read_best_ema_from_log(log_path)
    if best_ema_wr > 0.0:
        print(f"Best EMA win rate from log: {best_ema_wr:.4f}")

    iter_hist, loss_hist, wr_hist = [], [], []
    ema_wr = None

    for iteration in range(start_iteration, args.iterations):

        # ---- generate training examples ---- #
        if not args.no_generate:
            all_examples = parallel_generate_examples(
                num_games=args.games_per_iteration,
                ab_depth=args.ab_depth,
                max_plies=args.max_plies,
                num_workers=args.generation_workers,
            )

            if not all_examples:
                print(f"Iteration {iteration + 1}: no examples generated; skipping")
                continue

            save_generation(data_dir, all_examples, iteration + 1)
        else:
            all_examples = []

        # ---- load rolling window ---- #
        boards, cards, values = load_rolling_window(data_dir, args.window_size)
        if boards is None:
            print(f"Iteration {iteration + 1}: no data in window; skipping")
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
            if args.gauntlet_valuenet_mode == "tree-batched":
                g_wr = run_gauntlet_tree_search_batched(
                    model=model,
                    train_device=device,
                    gauntlet_device=gauntlet_device,
                    heuristic_depth=args.gauntlet_heuristic_depth,
                    search_depth=args.gauntlet_valuenet_depth,
                    max_plies=args.gauntlet_plies,
                    num_games=args.gauntlet_games,
                    progress_desc=f"Iter {iteration + 1} gauntlet(tree-batched)",
                )
            else:
                g_wr = run_gauntlet(
                    model=model,
                    heuristic_depth=args.gauntlet_heuristic_depth,
                    valuenet_depth=args.gauntlet_valuenet_depth,
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
            save_training_plot(iter_hist, loss_hist, wr_hist, plot_path,
                               log_scale=args.log_scale)

        save_checkpoint(save_path, model, optimizer, iteration + 1, args)

    print(f"Training complete. Checkpoint saved to {save_path}")


if __name__ == "__main__":
    main()
