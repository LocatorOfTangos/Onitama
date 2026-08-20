"""
VN1_train3.py – Stage-3 self-play training for ValueNet1

AlphaZero-style self-play where MCTS uses ValueNet to evaluate expanded leaf
nodes directly (value-only; no policy head).

Core design goals:
- Reuse Stage-2 training backbone (rolling window, augmentation, training loop,
  checkpoints, logging, plotting).
- Keep generation fast via multiprocessing and batched leaf inference.
- Use C++ opponents for gauntlet when available.
"""

import argparse
import csv
import math
import random
from pathlib import Path

import torch
import torch.multiprocessing as mp
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
from onitama.opponents.NN.VN1_train2 import (
    save_generation,
    load_rolling_window,
    train_epoch,
    init_csv_log,
    append_csv_log,
    read_best_ema_from_log,
    save_training_plot,
    save_checkpoint,
)

try:
    from onitama.engine.onitama_engine import (
        BoardState as CppBoardState,
        ABSearchBot as CppABSearchBot,
        MCTSBot as CppMCTSBot,
        WinColour as CppWinColour,
        set_mcts_python_compat as cpp_set_mcts_python_compat,
        ValueLeafMCTSTree as CppValueLeafMCTSTree,
    )
    import numpy as np
    _CPP_ENGINE = True
except ImportError:
    _CPP_ENGINE = False


# ------------------------------------------------------------------ #
#  Board-type-agnostic helpers (C++ BoardState vs Python Boardstate)   #
# ------------------------------------------------------------------ #

def _is_cpp_board(board):
    return _CPP_ENGINE and isinstance(board, CppBoardState)


def _board_copy(board):
    return board.copy()


def _board_get_moves(board):
    """Return a flat list of legal moves, captures first, instant-win short-circuit."""
    if _is_cpp_board(board):
        return board.possible_moves_captures_first()
    return board.possible_moves_search_optimised()


def _board_is_terminal(board):
    if _is_cpp_board(board):
        return board.is_terminal()
    return board.is_won() is not None


def _board_apply_move(board, move):
    """Apply move in-place and return undo record."""
    return board.apply_move(move)


def _board_terminal_value_current_player(board):
    """Return +1/-1 if terminal from current player's perspective, else None."""
    if _is_cpp_board(board):
        win = board.is_won()
        if not win:
            return None
        current_is_red = board.is_red_turn()
        if win.colour == CppWinColour.RED:
            return 1.0 if current_is_red else -1.0
        else:
            return -1.0 if current_is_red else 1.0
    terminal = board.is_won()
    if terminal is None:
        return None
    winner = terminal[0]
    return 1.0 if winner == board.turn_colour() else -1.0


def _encode_board(board):
    """Encode board for NN input. Returns (board_tensor, cards_tensor)."""
    if _is_cpp_board(board):
        b_np, c_np = board.encode_for_nn()
        return (
            torch.from_numpy(np.ascontiguousarray(b_np)),
            torch.from_numpy(np.ascontiguousarray(c_np)),
        )
    return encode_state_for_value_net(board)


class ValueLeafMCTSNode:
    """MCTS node that works with both Python Boardstate and C++ BoardState."""

    def __init__(self, board, parent=None, move=None):
        self.board = _board_copy(board)
        self.parent = parent
        self.move = move
        self.children = []
        self.untried_moves = _board_get_moves(self.board)
        self.visits = 0
        self.value_sum = 0.0

    def is_terminal(self):
        return _board_is_terminal(self.board)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def mean_value(self):
        if self.visits <= 0:
            return 0.0
        return self.value_sum / self.visits


# ---------------------------------------------------------------------------
# Pure-Python sequential MCTS helpers (fallback when C++ engine unavailable)
# ---------------------------------------------------------------------------

def _py_select_expand(root, exploration):
    """Select a leaf via UCB and expand one untried move. No virtual loss."""
    node = root
    while not node.is_terminal() and node.is_fully_expanded() and node.children:
        parent_visits = max(1, node.visits)

        def ucb(child, _pv=parent_visits):
            if child.visits == 0:
                return float("inf")
            exploitation = -(child.value_sum / child.visits)
            exploration_term = exploration * math.sqrt(
                math.log(_pv) / child.visits
            )
            return exploitation + exploration_term

        node = max(node.children, key=ucb)

    if not node.is_terminal() and not node.is_fully_expanded():
        move = node.untried_moves.pop()
        child_board = node.board.copy()
        child_board.apply_move(move)
        child = ValueLeafMCTSNode(child_board, parent=node, move=move)
        node.children.append(child)
        node = child

    return node


def _py_backpropagate(node, value):
    current = node
    signed_value = float(value)
    while current is not None:
        current.visits += 1
        current.value_sum += signed_value
        signed_value = -signed_value
        current = current.parent


# ---------------------------------------------------------------------------
# Proxy so _pick_root_move works with either Python root or C++ tree result
# ---------------------------------------------------------------------------

class _ChildProxy:
    """Lightweight stand-in for ValueLeafMCTSNode children used by _pick_root_move."""
    __slots__ = ("move", "visits")

    def __init__(self, move, visits):
        self.move   = move
        self.visits = visits


class _CppTreeRootProxy:
    """Wraps CppValueLeafMCTSTree so _pick_root_move sees a .children list."""
    __slots__ = ("children",)

    def __init__(self, cpp_tree):
        self.children = [
            _ChildProxy(move, visits)
            for move, visits in cpp_tree.root_child_visits()
        ]


# ---------------------------------------------------------------------------
# Central MCTS runner
#   - Fast path: C++ tree + single CPU forward pass per simulation
#   - Fallback: pure Python, same sequential design (no batching, no virtual loss)
# ---------------------------------------------------------------------------

def run_value_leaf_mcts(
    board,
    model,
    device,
    num_simulations,
    exploration,
    eval_batch_size=1,  # kept for API compat, ignored
    existing_tree=None,  # CppValueLeafMCTSTree to reuse (C++ path only)
):
    """Run value-leaf MCTS.

    Returns (root_proxy, root_value, tree) where:
      - root_proxy exposes .children with .move and .visits so _pick_root_move
        works unchanged.
      - tree is the live CppValueLeafMCTSTree (or None on the Python path).
        Pass it back as `existing_tree` on the next call after advancing it
        with tree.advance_to_child(move) to reuse all prior search work.
    """
    # ----------------------------------------------------------------
    # Fast path: C++ tree handles all select/expand/backprop overhead.
    # Python only runs the NN forward pass (one board at a time, CPU).
    # ----------------------------------------------------------------
    if _CPP_ENGINE and _is_cpp_board(board):
        if existing_tree is not None:
            tree = existing_tree
        else:
            tree = CppValueLeafMCTSTree(board, exploration)
        board_buf  = torch.empty(1, 4, 5, 5)
        cards_buf  = torch.empty(1, 3, 16)

        # Hoist no_grad context and get numpy views once; avoids per-sim overhead
        with torch.no_grad():
            board_view = board_buf.numpy()
            cards_view = cards_buf.numpy()
            for _ in range(num_simulations):
                node_idx, is_terminal, terminal_value, b_np, c_np = \
                    tree.select_and_expand()

                if is_terminal:
                    value = terminal_value
                else:
                    # Zero-copy: write directly into the tensor's backing memory
                    board_view[0] = b_np
                    cards_view[0] = c_np
                    value = float(model(board_buf, cards_buf).item())

                tree.backpropagate(node_idx, value)

        root_value = tree.root_value()
        return _CppTreeRootProxy(tree), root_value, tree

    # ----------------------------------------------------------------
    # Pure-Python fallback: sequential, no batching, no virtual loss.
    # ----------------------------------------------------------------
    root = ValueLeafMCTSNode(board)

    for _ in range(num_simulations):
        node = _py_select_expand(root, exploration)

        if node.is_terminal():
            value = _board_terminal_value_current_player(node.board) or 0.0
        else:
            if not hasattr(node, "_encoded"):
                node._encoded = _encode_board(node.board)
            bt, ct = node._encoded
            with torch.no_grad():
                value = float(
                    model(bt.unsqueeze(0).to(device),
                          ct.unsqueeze(0).to(device)).item()
                )

        _py_backpropagate(node, value)

    root_value = root.value_sum / root.visits if root.visits > 0 else 0.0
    return root, root_value, None


def _pick_root_move(root, temperature):
    """Select move proportional to visit counts (AlphaZero-style).

    temperature > 0: sample proportional to visits^(1/temperature)
    temperature <= 0: greedy (most visits)
    """
    if not root.children:
        return None
    if len(root.children) == 1:
        return root.children[0].move

    visits = [c.visits for c in root.children]

    if temperature <= 0:
        best_idx = max(range(len(root.children)), key=lambda i: visits[i])
        return root.children[best_idx].move

    # Exponentiated visit counts
    inv_temp = 1.0 / temperature
    weighted = [max(v, 0) ** inv_temp for v in visits]
    z = sum(weighted)
    if z == 0:
        return random.choice(root.children).move
    probs = [w / z for w in weighted]

    r = random.random()
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if r <= c:
            return root.children[i].move
    return root.children[-1].move


def _play_stage3_self_play_game(
    model,
    device,
    mcts_simulations,
    exploration,
    temperature,
    eval_batch_size,
    max_plies,
):
    # Draw a fresh board seed from Python's random state so successive games
    # within the same worker are not deterministically correlated.
    if _CPP_ENGINE:
        board = CppBoardState.random_start(random.getrandbits(32))
    else:
        board = Boardstate.Boardstate()
    examples = []
    tree = None  # live C++ tree carried across plies for reuse

    for _ in range(max_plies):
        if _board_is_terminal(board):
            break

        root, root_value, tree = run_value_leaf_mcts(
            board=board,
            model=model,
            device=device,
            num_simulations=mcts_simulations,
            exploration=exploration,
            eval_batch_size=eval_batch_size,
            existing_tree=tree,
        )

        bt, ct = _encode_board(board)
        examples.append((bt, ct, root_value))

        move = _pick_root_move(root, temperature)
        if move is None:
            break
        board.apply_move(move)

        # Advance the tree to the chosen child so its subtree is retained
        # for the next MCTS run.  Fall back to a fresh tree if that child
        # was never expanded (exploration budget was too small to reach it).
        if tree is not None:
            if not tree.advance_to_child(move):
                tree = None  # child not in tree; create fresh next iteration

    return [(bt.clone(), ct.clone(), v) for bt, ct, v in examples]


def _training_worker(args_tuple):
    (
        state_dict,
        num_games,
        mcts_simulations,
        exploration,
        temperature,
        eval_batch_size,
        max_plies,
        seed,
    ) = args_tuple

    random.seed(seed)
    torch.manual_seed(seed)

    model = ValueNet1()
    model.load_state_dict(state_dict)
    model.eval()

    # One thread per worker: avoids CPU over-subscription with many parallel workers.
    torch.set_num_threads(1)
    device = torch.device("cpu")

    # JIT-trace for faster single-sample inference (~2x on CPU batch_size=1)
    try:
        _trace_b = torch.empty(1, 4, 5, 5)
        _trace_c = torch.empty(1, 3, 16)
        model = torch.jit.trace(model, (_trace_b, _trace_c))
        model.eval()
    except Exception:
        pass  # fall back to regular model if tracing fails

    all_examples = []

    for _ in range(num_games):
        # Do NOT re-seed here: the worker's random state was seeded once at
        # startup (above), giving diversity across workers.  Re-seeding per
        # game would cause every training iteration to replay the exact same
        # set of starting positions for a given worker.
        all_examples.extend(
            _play_stage3_self_play_game(
                model=model,
                device=device,
                mcts_simulations=mcts_simulations,
                exploration=exploration,
                temperature=temperature,
                eval_batch_size=eval_batch_size,
                max_plies=max_plies,
            )
        )

    return all_examples


def parallel_stage3_self_play(
    model,
    num_games,
    mcts_simulations,
    exploration,
    temperature,
    eval_batch_size,
    max_plies,
    seed,
    num_workers=None,
):
    if num_games <= 0:
        return []

    if num_workers is None:
        num_workers = min(num_games, mp.cpu_count())
    num_workers = min(max(1, num_workers), num_games)

    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    games_per_worker = [num_games // num_workers] * num_workers
    for i in range(num_games % num_workers):
        games_per_worker[i] += 1

    worker_args = []
    for i, games_i in enumerate(games_per_worker):
        worker_args.append(
            (
                state_dict,
                games_i,
                mcts_simulations,
                exploration,
                temperature,
                eval_batch_size,
                max_plies,
                seed + i,
            )
        )

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_training_worker, worker_args),
                total=len(worker_args),
                desc=f"Stage-3 self-play ({num_workers} workers)",
                leave=False,
            )
        )

    out = []
    for part in results:
        out.extend(part)
    return out


def _cpp_request_opponent_move(opponent_kind, cpp_board, kwargs):
    if opponent_kind == "ab":
        opponent = CppABSearchBot(kwargs.get("depth", 4))
        return opponent.request_move(cpp_board, -1.0)

    if opponent_kind == "mcts":
        time_ms = int(kwargs.get("time_limit", 0.5) * 1000)
        threads = int(kwargs.get("threads", max(1, mp.cpu_count() // 2)))
        cpp_set_mcts_python_compat(kwargs.get("python_compat", False))
        bot = CppMCTSBot(time_ms=time_ms, threads=threads)
        return bot.request_move(cpp_board)

    if opponent_kind == "random":
        moves = cpp_board.possible_moves()
        return random.choice(moves) if moves else None

    raise ValueError(f"Unknown opponent kind: {opponent_kind}")


def _python_request_opponent_move(opponent_kind, board, kwargs):
    if opponent_kind == "ab":
        bot = ab_search_bot.ABSearchBot(max_depth=kwargs.get("depth", 4))
    elif opponent_kind == "mcts":
        bot = MCTSBot(time_limit=kwargs.get("time_limit", 0.5))
    elif opponent_kind == "random":
        bot = RandomBot()
    else:
        raise ValueError(f"Unknown opponent kind: {opponent_kind}")
    return bot.request_move(board)


def _gauntlet_game_worker(args_tuple):
    (
        state_dict,
        mcts_simulations,
        exploration,
        eval_batch_size,
        max_plies,
        opponent_kind,
        opponent_kwargs,
        seed,
    ) = args_tuple

    random.seed(seed)
    torch.manual_seed(seed)

    model = ValueNet1()
    model.load_state_dict(state_dict)
    model.eval()
    torch.set_num_threads(1)
    device = torch.device("cpu")

    # JIT-trace for faster single-sample inference
    try:
        _trace_b = torch.empty(1, 4, 5, 5)
        _trace_c = torch.empty(1, 3, 16)
        model = torch.jit.trace(model, (_trace_b, _trace_c))
        model.eval()
    except Exception:
        pass

    if _CPP_ENGINE:
        cpp_board = CppBoardState.random_start(seed)
        tree = None  # live tree carried across plies for reuse
        for _ in range(max_plies):
            if cpp_board.is_terminal():
                break

            if cpp_board.is_red_turn():
                # MCTS runs directly on C++ board — no py↔cpp conversion
                root, _, tree = run_value_leaf_mcts(
                    board=cpp_board,
                    model=model,
                    device=device,
                    num_simulations=mcts_simulations,
                    exploration=exploration,
                    eval_batch_size=eval_batch_size,
                    existing_tree=tree,
                )
                move = _pick_root_move(root, temperature=0.0)
                if move is None:
                    break
                cpp_board.apply_move(move)
                # Advance through our move
                if tree is not None and not tree.advance_to_child(move):
                    tree = None
            else:
                move = _cpp_request_opponent_move(opponent_kind, cpp_board, opponent_kwargs)
                if move is None or (hasattr(move, "valid") and not move.valid()):
                    break
                cpp_board.apply_move(move)
                # Advance through opponent's move so we keep the explored subtree
                if tree is not None and not tree.advance_to_child(move):
                    tree = None

        win = cpp_board.is_won()
        if win and win.colour == CppWinColour.RED:
            return 1.0
        if not win:
            return 0.5
        return 0.0

    board = Boardstate.Boardstate()
    for _ in range(max_plies):
        if board.is_won() is not None:
            break

        if board.turn_colour() == "RED":
            root, _ = run_value_leaf_mcts(
                board=board,
                model=model,
                device=device,
                num_simulations=mcts_simulations,
                exploration=exploration,
                eval_batch_size=eval_batch_size,
            )
            move = _pick_root_move(root, temperature=0.0)
        else:
            move = _python_request_opponent_move(opponent_kind, board, opponent_kwargs)

        if move is None:
            break
        board.apply_move(move)

    terminal = board.is_won()
    if terminal is None:
        return 0.5
    return 1.0 if terminal[0] == "RED" else 0.0


def run_stage3_gauntlet(
    model,
    mcts_simulations,
    exploration,
    eval_batch_size,
    max_plies,
    num_games,
    opponent_kind,
    opponent_kwargs,
    num_workers=None,
    progress_desc=None,
):
    if num_games <= 0:
        return None

    if num_workers is None:
        num_workers = min(num_games, mp.cpu_count())
    num_workers = min(max(1, num_workers), num_games)

    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    worker_args = [
        (
            state_dict,
            mcts_simulations,
            exploration,
            eval_batch_size,
            max_plies,
            opponent_kind,
            opponent_kwargs,
            i,
        )
        for i in range(num_games)
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_gauntlet_game_worker, worker_args),
                total=num_games,
                desc=progress_desc or "Stage-3 gauntlet",
                leave=False,
            )
        )

    return sum(results) / num_games


def regenerate_window(
    model,
    data_dir,
    window_size,
    games_per_generation,
    mcts_simulations,
    exploration,
    temperature,
    eval_batch_size,
    max_plies,
    seed,
    num_workers=None,
):
    existing = sorted(data_dir.glob("gen_*.pt")) if data_dir.exists() else []
    for f in existing:
        f.unlink()

    data_dir.mkdir(parents=True, exist_ok=True)

    for gen_idx in range(window_size):
        examples = parallel_stage3_self_play(
            model=model,
            num_games=games_per_generation,
            mcts_simulations=mcts_simulations,
            exploration=exploration,
            temperature=temperature,
            eval_batch_size=eval_batch_size,
            max_plies=max_plies,
            seed=seed - window_size + gen_idx,
            num_workers=num_workers,
        )
        if examples:
            save_generation(data_dir, examples, gen_idx + 1)
        print(
            f"Window generation {gen_idx + 1}/{window_size}: {len(examples)} examples"
        )


def _load_training_history_from_log(path, upto_iteration=None):
    """Load existing CSV history for seamless resume plotting/EMA continuity."""
    iter_hist, loss_hist, wr_hist = [], [], []
    last_ema = None

    if not path.exists():
        return iter_hist, loss_hist, wr_hist, last_ema

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iter_str = (row.get("iteration") or "").strip()
            loss_str = (row.get("avg_loss") or "").strip()
            if not iter_str or not loss_str:
                continue

            try:
                iteration = int(float(iter_str))
                avg_loss = float(loss_str)
            except ValueError:
                continue

            if upto_iteration is not None and iteration > upto_iteration:
                continue

            ema_str = (row.get("ema_win_rate") or "").strip()
            ema_val = None
            if ema_str:
                try:
                    ema_val = float(ema_str)
                    last_ema = ema_val
                except ValueError:
                    ema_val = None

            iter_hist.append(iteration)
            loss_hist.append(avg_loss)
            wr_hist.append(ema_val)

    return iter_hist, loss_hist, wr_hist, last_ema


def main():
    parser = argparse.ArgumentParser(
        description="VN1 Stage-3: ValueLeaf-MCTS self-play training",
    )

    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--self-games-per-iter", type=int, default=48)
    parser.add_argument("--max-plies", type=int, default=180)

    parser.add_argument("--mcts-simulations", type=int, default=512)
    parser.add_argument("--mcts-exploration", type=float, default=1.4)
    parser.add_argument("--self-play-temperature", type=float, default=0.50) # 0.10 later
    parser.add_argument("--eval-batch-size", type=int, default=64)

    parser.add_argument("--epochs-per-iteration", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true", default=True)

    parser.add_argument("--self-play-workers", type=int, default=None)

    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default="logs3/generations")
    parser.add_argument("--no-flip", action="store_true")

    parser.add_argument("--gauntlet-every", type=int, default=4)
    parser.add_argument("--gauntlet-games", type=int, default=32)
    parser.add_argument("--gauntlet-opponent", type=str, default="ab", choices=["ab", "mcts", "random"])
    parser.add_argument("--gauntlet-plies", type=int, default=180)
    parser.add_argument("--gauntlet-ab-depth", type=int, default=5)
    parser.add_argument("--gauntlet-mcts-time-limit", type=float, default=0.5)
    parser.add_argument("--gauntlet-mcts-threads", type=int, default=None)
    parser.add_argument("--gauntlet-mcts-python-compat", action="store_true", default=False)

    parser.add_argument("--save-path", type=str, default="logs3/vn1_stage3_checkpoint.pt")
    parser.add_argument("--load-checkpoint", type=str, default=None)
    parser.add_argument("--plot-path", type=str, default="logs3/vn1_stage3_loss_plot.png")
    parser.add_argument("--plot-every", type=int, default=1)
    parser.add_argument("--log-scale", action="store_true", default=False,
                        help="Use logarithmic scale for loss in the training plot")
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--regenerate-window", action="store_true", default=False)

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
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.MSELoss()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    start_iteration = 0

    if args.resume and save_path.exists():
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iteration = ckpt.get("iteration", 0)
        print(f"Resumed from {save_path} at iteration {start_iteration}")
    elif args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded weights from {args.load_checkpoint}")

    data_dir = Path(args.data_dir)
    flip_map = build_card_flip_map()

    if args.regenerate_window:
        regenerate_window(
            model=model,
            data_dir=data_dir,
            window_size=args.window_size,
            games_per_generation=args.self_games_per_iter,
            mcts_simulations=args.mcts_simulations,
            exploration=args.mcts_exploration,
            temperature=args.self_play_temperature,
            eval_batch_size=args.eval_batch_size,
            max_plies=args.max_plies,
            seed=args.seed,
            num_workers=args.self_play_workers,
        )

    default_log = Path(__file__).resolve().parent / "logs3" / "vn1_train3_log.csv"
    log_path = Path(args.log_path) if args.log_path else default_log
    init_csv_log(log_path)

    plot_path = Path(args.plot_path)
    half_life = 4

    if args.resume:
        iter_hist, loss_hist, wr_hist, ema_wr = _load_training_history_from_log(
            log_path,
            upto_iteration=start_iteration if start_iteration > 0 else None,
        )
        if iter_hist:
            print(
                f"Loaded {len(iter_hist)} prior log rows from {log_path} "
                f"(through iteration {iter_hist[-1]})"
            )
    else:
        iter_hist, loss_hist, wr_hist, ema_wr = [], [], [], None

    pb_save_path = Path(args.save_path).parent / "vn1_stage3_pb_checkpoint.pt"
    best_ema_wr = read_best_ema_from_log(log_path)

    gauntlet_kwargs = {}
    if args.gauntlet_opponent == "ab":
        gauntlet_kwargs["depth"] = args.gauntlet_ab_depth
    elif args.gauntlet_opponent == "mcts":
        gauntlet_kwargs["time_limit"] = args.gauntlet_mcts_time_limit
        if args.gauntlet_mcts_threads is not None:
            gauntlet_kwargs["threads"] = args.gauntlet_mcts_threads
        gauntlet_kwargs["python_compat"] = args.gauntlet_mcts_python_compat

    for iteration in range(start_iteration, args.iterations):
        random.seed(args.seed + iteration)

        all_examples = parallel_stage3_self_play(
            model=model,
            num_games=args.self_games_per_iter,
            mcts_simulations=args.mcts_simulations,
            exploration=args.mcts_exploration,
            temperature=args.self_play_temperature,
            eval_batch_size=args.eval_batch_size,
            max_plies=args.max_plies,
            seed=args.seed + iteration * max(1, args.self_games_per_iter),
            num_workers=args.self_play_workers,
        )

        if not all_examples:
            print(f"Iteration {iteration + 1}: no examples generated; skipping")
            continue

        save_generation(data_dir, all_examples, iteration + 1)

        boards, cards, values = load_rolling_window(data_dir, args.window_size)
        if boards is None:
            continue

        if not args.no_flip:
            boards, cards, values = augment_with_horizontal_flip(
                boards, cards, values, flip_map
            )

        losses = []
        for epoch in range(args.epochs_per_iteration):
            desc = f"Iter {iteration + 1} Epoch {epoch + 1}/{args.epochs_per_iteration}"
            loss = train_epoch(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                boards=boards,
                cards=cards,
                values=values,
                batch_size=args.batch_size,
                device=device,
                progress_desc=desc,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers,
            )
            losses.append(loss)

        avg_loss = sum(losses) / max(1, len(losses))

        g_wr = None
        if args.gauntlet_every > 0 and (iteration + 1) % args.gauntlet_every == 0:
            g_wr = run_stage3_gauntlet(
                model=model,
                mcts_simulations=args.mcts_simulations,
                exploration=args.mcts_exploration,
                eval_batch_size=args.eval_batch_size,
                max_plies=args.gauntlet_plies,
                num_games=args.gauntlet_games,
                opponent_kind=args.gauntlet_opponent,
                opponent_kwargs=gauntlet_kwargs,
                num_workers=args.self_play_workers,
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
