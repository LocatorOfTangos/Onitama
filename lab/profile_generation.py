"""
profile_generation.py — In-process profiler for Stage-3 self-play data generation.

Two complementary views:
  1. cProfile  — function-level call counts and cumulative time, sorted by
                 tottime.  Ideal for spotting unexpected hot functions.
  2. Manual timing — per-simulation breakdown of the four phases inside the
                 MCTS hot loop:
                   • select_and_expand  (C++ tree traversal + encoding)
                   • NN forward pass    (ValueNet1 inference, CPU)
                   • backpropagate      (C++ tree update)
                   • advance_to_child   (C++ compaction, called once per ply)
                 plus board-encoding and move-picking overhead per ply.
                 Prints mean ± std over all simulations/plies in the run.

Usage (from the NN directory, or anywhere with the package on the path):

    # Quick run — 1 game, default 512 sims
    python lab/profile_generation.py

    # Match your real training settings
    python lab/profile_generation.py --games 3 --sims 2048 --cprofile-lines 40

    # Skip cProfile, only get timing breakdown
    python lab/profile_generation.py --no-cprofile --games 5

The script loads the latest checkpoint from logs3/ automatically (or you can
specify --checkpoint).
"""

import argparse
import cProfile
import io
import math
import pstats
import random
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Resolve package root
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from onitama.opponents.NN.architecture.ValueNet1 import ValueNet1

try:
    from onitama.engine.onitama_engine import (
        ValueLeafMCTSTree as CppValueLeafMCTSTree,
        BoardState as CppBoardState,
        WinColour as CppWinColour,
    )
    _CPP_ENGINE = True
except ImportError:
    _CPP_ENGINE = False
    print("WARNING: C++ engine not available; timings will reflect the Python fallback.")

# Import the full trainer module so we can reuse its functions
from onitama.opponents.NN.VN1_train3 import (
    _training_worker,
    _play_stage3_self_play_game,
    run_value_leaf_mcts,
    _pick_root_move,
    _CppTreeRootProxy,
    _encode_board,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path):
    model = ValueNet1()
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found — using random weights.")
    model.eval()
    # JIT-trace, same as the worker
    try:
        model = torch.jit.trace(model, (torch.empty(1, 4, 5, 5), torch.empty(1, 3, 16)))
        model.eval()
        print("JIT trace: OK")
    except Exception as e:
        print(f"JIT trace failed ({e}); using eager model.")
    return model


def _find_latest_checkpoint(default_dir: Path) -> str | None:
    candidates = [
        default_dir / "vn1_stage3_checkpoint.pt",
        default_dir / "vn1_stage3_pb_checkpoint.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


# ---------------------------------------------------------------------------
# Section 1: cProfile wrapper
# ---------------------------------------------------------------------------

def run_cprofile(state_dict, num_games, sims, exploration, temperature,
                 max_plies, top_n, seed):
    args_tuple = (
        state_dict, num_games, sims, exploration, temperature,
        1,          # eval_batch_size (ignored)
        max_plies,
        seed,
    )

    pr = cProfile.Profile()
    pr.enable()
    _training_worker(args_tuple)
    pr.disable()

    buf = io.StringIO()
    ps  = pstats.Stats(pr, stream=buf).sort_stats("tottime")
    ps.print_stats(top_n)
    print("\n" + "=" * 70)
    print("cPROFILE  (sorted by tottime, top %d functions)" % top_n)
    print("=" * 70)
    print(buf.getvalue())


# ---------------------------------------------------------------------------
# Section 2: Manual per-phase timing
# ---------------------------------------------------------------------------

def _timed_run_value_leaf_mcts(board, model, num_simulations, exploration,
                                existing_tree, phase_accum):
    """
    Drop-in replacement for run_value_leaf_mcts that accumulates per-phase
    wall-clock times into `phase_accum` dict.
    """
    if not _CPP_ENGINE:
        # Fall back to untimed call so the rest of the script still works
        return run_value_leaf_mcts(board, model, "cpu", num_simulations,
                                   exploration, existing_tree=existing_tree)

    if existing_tree is not None:
        tree = existing_tree
    else:
        tree = CppValueLeafMCTSTree(board, exploration)

    board_buf = torch.empty(1, 4, 5, 5)
    cards_buf = torch.empty(1, 3, 16)

    with torch.no_grad():
        board_view = board_buf.numpy()
        cards_view = cards_buf.numpy()

        for _ in range(num_simulations):
            # --- select_and_expand ---
            t0 = time.perf_counter()
            node_idx, is_terminal, terminal_value, b_np, c_np = tree.select_and_expand()
            t1 = time.perf_counter()
            phase_accum["select_expand"].append(t1 - t0)

            # --- NN forward ---
            t0 = time.perf_counter()
            if is_terminal:
                value = terminal_value
            else:
                board_view[0] = b_np
                cards_view[0] = c_np
                value = float(model(board_buf, cards_buf).item())
            t1 = time.perf_counter()
            if not is_terminal:
                phase_accum["nn_forward"].append(t1 - t0)
            else:
                phase_accum["terminal_shortcut"].append(t1 - t0)

            # --- backpropagate ---
            t0 = time.perf_counter()
            tree.backpropagate(node_idx, value)
            t1 = time.perf_counter()
            phase_accum["backpropagate"].append(t1 - t0)

    root_value = tree.root_value()
    return _CppTreeRootProxy(tree), root_value, tree


def run_timed(model, num_games, sims, exploration, temperature, max_plies):
    """Run num_games and collect per-phase timing stats."""
    phase_accum = {
        "select_expand":     [],
        "nn_forward":        [],
        "terminal_shortcut": [],
        "backpropagate":     [],
        "advance_to_child":  [],
        "encode_board":      [],
        "pick_move":         [],
    }
    ply_totals = []   # wall time per ply (end-to-end)
    game_totals = []  # wall time per game

    torch.set_num_threads(1)
    device = torch.device("cpu")

    for g in range(num_games):
        board = CppBoardState.random_start(random.getrandbits(32)) if _CPP_ENGINE \
                else None
        tree  = None
        t_game = time.perf_counter()

        for _ in range(max_plies):
            if _CPP_ENGINE and board.is_terminal():
                break
            t_ply = time.perf_counter()

            # MCTS
            root, root_value, tree = _timed_run_value_leaf_mcts(
                board, model, sims, exploration, tree, phase_accum
            )

            # Encode board for training example
            t0 = time.perf_counter()
            _encode_board(board)
            phase_accum["encode_board"].append(time.perf_counter() - t0)

            # Pick move
            t0 = time.perf_counter()
            move = _pick_root_move(root, temperature)
            phase_accum["pick_move"].append(time.perf_counter() - t0)

            if move is None:
                break
            board.apply_move(move)

            # Advance / compact tree
            if tree is not None:
                t0 = time.perf_counter()
                if not tree.advance_to_child(move):
                    tree = None
                phase_accum["advance_to_child"].append(time.perf_counter() - t0)

            ply_totals.append(time.perf_counter() - t_ply)

        game_totals.append(time.perf_counter() - t_game)

    # --- Report ---
    def _stats(lst):
        if not lst:
            return 0.0, 0.0, 0
        n  = len(lst)
        mu = sum(lst) / n
        std = math.sqrt(sum((x - mu) ** 2 for x in lst) / n) if n > 1 else 0.0
        return mu * 1e6, std * 1e6, n   # convert to µs

    nn_calls   = len(phase_accum["nn_forward"])
    term_calls = len(phase_accum["terminal_shortcut"])
    total_sims = nn_calls + term_calls

    print("\n" + "=" * 70)
    print("MANUAL TIMING BREAKDOWN")
    print("=" * 70)
    print(f"  Games:              {num_games}")
    print(f"  Plies collected:    {len(ply_totals)}")
    print(f"  Total simulations:  {total_sims}  "
          f"(NN evals: {nn_calls}, terminal shortcuts: {term_calls})")
    if nn_calls:
        pct_term = 100.0 * term_calls / total_sims
        print(f"  Terminal rate:      {pct_term:.1f}%  (shortcuts save NN call)")
    print()

    rows = [
        ("select_and_expand",   phase_accum["select_expand"]),
        ("nn_forward (non-term)", phase_accum["nn_forward"]),
        ("terminal_shortcut",   phase_accum["terminal_shortcut"]),
        ("backpropagate",        phase_accum["backpropagate"]),
        ("advance_to_child",     phase_accum["advance_to_child"]),
        ("encode_board (ply)",   phase_accum["encode_board"]),
        ("pick_move (ply)",      phase_accum["pick_move"]),
    ]

    # Header
    print(f"  {'Phase':<28}  {'mean (µs)':>11}  {'std (µs)':>10}  {'calls':>8}  {'total (ms)':>12}")
    print("  " + "-" * 75)
    total_accounted = 0.0
    for label, lst in rows:
        mu, std, n = _stats(lst)
        total_ms = mu * n / 1e3
        total_accounted += total_ms
        print(f"  {label:<28}  {mu:>11.2f}  {std:>10.2f}  {n:>8}  {total_ms:>11.1f}")

    print()
    mu_ply, std_ply, n_ply = _stats(ply_totals)
    print(f"  {'ply wall-clock':<28}  {mu_ply:>11.2f}  {std_ply:>10.2f}  {n_ply:>8}")
    print()

    # Budget breakdown: of the average ply time, how much is each phase?
    if ply_totals and phase_accum["nn_forward"]:
        avg_ply_us   = sum(ply_totals) / len(ply_totals) * 1e6
        avg_sel_us   = sum(phase_accum["select_expand"]) / len(phase_accum["select_expand"]) * 1e6
        avg_nn_us    = sum(phase_accum["nn_forward"]) / len(phase_accum["nn_forward"]) * 1e6
        avg_bp_us    = sum(phase_accum["backpropagate"]) / len(phase_accum["backpropagate"]) * 1e6
        nn_total_us  = avg_nn_us * nn_calls
        sel_total_us = avg_sel_us * len(phase_accum["select_expand"])
        bp_total_us  = avg_bp_us * len(phase_accum["backpropagate"])
        ply_total_us = avg_ply_us * len(ply_totals)

        print("  Budget per ply (% of ply wall-clock):")
        for name, val in [
            ("select_and_expand × sims", sel_total_us),
            ("nn_forward × nn-sims",      nn_total_us),
            ("backpropagate × sims",      bp_total_us),
        ]:
            print(f"    {name:<34}  {100 * val / ply_total_us:>5.1f}%")
        print()

    print(f"  Game wall-clock:  mean {sum(game_totals)/len(game_totals):.2f}s  "
          f"total {sum(game_totals):.2f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile Stage-3 self-play data generation."
    )
    parser.add_argument("--games",       type=int,   default=1,
                        help="Number of games to profile (default: 1)")
    parser.add_argument("--sims",        type=int,   default=512,
                        help="MCTS simulations per ply (default: 512)")
    parser.add_argument("--exploration", type=float, default=1.4)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--max-plies",   type=int,   default=180)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--checkpoint",  type=str,   default=None,
                        help="Path to .pt checkpoint (auto-detected if omitted)")
    parser.add_argument("--no-cprofile", action="store_true",
                        help="Skip cProfile section (faster)")
    parser.add_argument("--cprofile-lines", type=int, default=30,
                        help="Number of top functions to print from cProfile")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    # Resolve checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None:
        log_dir = Path(__file__).resolve().parent.parent / "src" / "onitama" / "opponents" / "NN" / "logs3"
        checkpoint = _find_latest_checkpoint(log_dir)

    model = _load_model(checkpoint)
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    print(f"\nProfiling: {args.games} game(s), {args.sims} sims/ply, "
          f"max {args.max_plies} plies, seed {args.seed}")
    print(f"C++ engine: {'YES' if _CPP_ENGINE else 'NO (Python fallback)'}\n")

    # --- Section 1: cProfile ---
    if not args.no_cprofile:
        run_cprofile(
            state_dict=state_dict,
            num_games=args.games,
            sims=args.sims,
            exploration=args.exploration,
            temperature=args.temperature,
            max_plies=args.max_plies,
            top_n=args.cprofile_lines,
            seed=args.seed,
        )

    # --- Section 2: Manual timing ---
    run_timed(
        model=model,
        num_games=args.games,
        sims=args.sims,
        exploration=args.exploration,
        temperature=args.temperature,
        max_plies=args.max_plies,
    )


if __name__ == "__main__":
    main()
