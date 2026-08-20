"""
analyze_expansion.py — Measure MCTS tree expansion patterns to evaluate
the viability of "batch all children on first expansion" as a speedup strategy.

For each MCTS position snapshot this script reports:

  • Total visited non-terminal nodes in the finished tree.
  • Fraction of those that are NOT FULLY EXPANDED (NFE) — i.e. still have
    untried moves.  A high fraction means most of the tree is still partially
    explored; a low fraction means the budget was large enough to exhaust most
    branches.

  • For NFE nodes only, the distribution of the EXPANSION RATIO:
        r = n_expanded_children / n_total_children
    where n_total = n_expanded + n_untried.
    r ≈ 0 means we have barely touched the node      → large future batch potential
    r ≈ 1 means almost all children are already tried → little left to batch

  • For NFE nodes, the distribution of N_UNTRIED (= n_total - n_expanded),
    which is exactly the batch size you would send to the NN if you evaluated
    all remaining children at once.  Higher is better for batching.

  • "Pristine" NFE nodes (n_expanded == 0, never been partially expanded before)
    vs "partial" NFE nodes (some children already visited).  Pristine nodes give
    you the full branching-factor as batch size.

  • Visits-weighted versions of the above metrics, reflecting how often each
    type of node is actually reached during search (not just its presence in
    the tree).

  • Projected NN-call savings under a "full expansion on first visit" strategy:
    if on every first expansion you evaluate all n_total children at once,
    subsequent visits to that node cost 0 NN calls.  We compute:
        savings% = (sims - unique_nodes_created) / sims × 100
    which is the fraction of NN calls that become redundant.

Usage (from the Onitama project root):

    python lab/analyze_expansion.py                          # quick defaults
    python lab/analyze_expansion.py --sims 128 512 2048      # multiple budgets
    python lab/analyze_expansion.py --positions 20 --sims 1024
    python lab/analyze_expansion.py --checkpoint path/to/ckpt.pt
"""

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from onitama.opponents.NN.architecture.ValueNet1 import ValueNet1

try:
    from onitama.engine.onitama_engine import (
        ValueLeafMCTSTree as CppValueLeafMCTSTree,
        BoardState as CppBoardState,
    )
    _CPP = True
except ImportError:
    _CPP = False
    print("ERROR: C++ engine required for this script.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: str | None) -> torch.nn.Module:
    model = ValueNet1()
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found – using random weights (patterns will still be valid).")
    model.eval()
    try:
        model = torch.jit.trace(model, (torch.empty(1, 4, 5, 5), torch.empty(1, 3, 16)))
        model.eval()
    except Exception:
        pass
    return model


def _find_checkpoint() -> str | None:
    d = _SRC / "onitama" / "opponents" / "NN" / "logs3"
    for name in ("vn1_stage3_checkpoint.pt", "vn1_stage3_pb_checkpoint.pt"):
        p = d / name
        if p.exists():
            return str(p)
    return None


def _run_mcts(board, model, num_sims: int, exploration: float) -> CppValueLeafMCTSTree:
    """Run num_sims simulations and return the finished tree (not advanced)."""
    tree = CppValueLeafMCTSTree(board, exploration)
    buf_b = torch.empty(1, 4, 5, 5)
    buf_c = torch.empty(1, 3, 16)
    with torch.no_grad():
        view_b = buf_b.numpy()
        view_c = buf_c.numpy()
        for _ in range(num_sims):
            idx, is_term, term_val, b_np, c_np = tree.select_and_expand()
            if is_term:
                value = term_val
            else:
                view_b[0] = b_np
                view_c[0] = c_np
                value = float(model(buf_b, buf_c).item())
            tree.backpropagate(idx, value)
    return tree


def _percentile(arr: np.ndarray, p: float) -> float:
    if len(arr) == 0:
        return 0.0
    return float(np.percentile(arr, p))


# ---------------------------------------------------------------------------
# Core analysis per snapshot
# ---------------------------------------------------------------------------

def analyse_tree(tree: CppValueLeafMCTSTree, num_sims: int) -> dict:
    """Extract expansion stats from a finished tree and return a results dict."""
    total_visited, n_nfe, n_exp_arr, n_tot_arr, vis_arr = tree.node_expansion_stats()

    n_exp_arr = np.asarray(n_exp_arr, dtype=np.int32)
    n_tot_arr = np.asarray(n_tot_arr, dtype=np.int32)
    vis_arr   = np.asarray(vis_arr,   dtype=np.int32)
    n_untr_arr = n_tot_arr - n_exp_arr   # untried children remaining

    # Expansion ratios
    ratios = np.where(n_tot_arr > 0, n_exp_arr / n_tot_arr.astype(float), 0.0)

    # Pristine = never had any child expanded yet
    pristine_mask = (n_exp_arr == 0)
    n_pristine  = int(pristine_mask.sum())
    n_partial   = n_nfe - n_pristine

    # Visits-weighted stats (weight each node by how often it is reached)
    total_vis = vis_arr.sum() if len(vis_arr) > 0 else 0
    w = vis_arr.astype(float) / total_vis if total_vis > 0 else np.ones(len(vis_arr)) / max(1, len(vis_arr))
    weighted_ratio   = float((ratios * w).sum()) if len(ratios) > 0 else 0.0
    weighted_n_untr  = float((n_untr_arr * w).sum()) if len(n_untr_arr) > 0 else 0.0
    weighted_n_tot   = float((n_tot_arr  * w).sum()) if len(n_tot_arr) > 0 else 0.0

    # Projected savings under "expand all children on first visit":
    # Under that policy, the number of NN calls = number of unique nodes created
    # (one batch per node, evaluating all children simultaneously).
    # Current cost = num_sims NN calls. Under batch-all = total_visited (approx).
    # Savings = how many NN calls become redundant.
    projected_nn_calls_batch = total_visited  # one NN call per unique node (batched)
    savings_pct = 100.0 * (num_sims - projected_nn_calls_batch) / num_sims if num_sims > 0 else 0.0

    return dict(
        num_sims          = num_sims,
        total_visited     = total_visited,
        n_nfe             = n_nfe,
        pct_nfe           = 100.0 * n_nfe / total_visited if total_visited > 0 else 0.0,
        n_pristine        = n_pristine,
        n_partial         = n_partial,
        pct_pristine_of_nfe = 100.0 * n_pristine / n_nfe if n_nfe > 0 else 0.0,
        # Unweighted distribution of expansion ratio
        ratio_mean        = float(ratios.mean()) if len(ratios) > 0 else 0.0,
        ratio_p25         = _percentile(ratios, 25),
        ratio_p50         = _percentile(ratios, 50),
        ratio_p75         = _percentile(ratios, 75),
        # Unweighted n_untried distribution (= pending batch size)
        untr_mean         = float(n_untr_arr.mean()) if len(n_untr_arr) > 0 else 0.0,
        untr_p25          = _percentile(n_untr_arr, 25),
        untr_p50          = _percentile(n_untr_arr, 50),
        untr_p75          = _percentile(n_untr_arr, 75),
        untr_max          = int(n_untr_arr.max()) if len(n_untr_arr) > 0 else 0,
        # Unweighted branching factor (total children)
        tot_mean          = float(n_tot_arr.mean()) if len(n_tot_arr) > 0 else 0.0,
        tot_p50           = _percentile(n_tot_arr, 50),
        # Visits-weighted versions of the key metrics
        w_ratio_mean      = weighted_ratio,
        w_n_untr_mean     = weighted_n_untr,
        w_n_tot_mean      = weighted_n_tot,
        # Projected savings
        projected_savings_pct = max(0.0, savings_pct),
        # Raw arrays for histogram printing
        _ratios    = ratios,
        _n_untr    = n_untr_arr,
        _n_tot     = n_tot_arr,
        _pristine  = pristine_mask,
    )


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _hist_str(arr: np.ndarray, bins, fmt=".0f") -> str:
    """One-line ASCII histogram."""
    if len(arr) == 0:
        return "(no data)"
    counts, edges = np.histogram(arr, bins=bins)
    total = counts.sum()
    parts = []
    for i, c in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        pct = 100.0 * c / total if total > 0 else 0.0
        label = f"[{lo:{fmt}},{hi:{fmt}})"
        parts.append(f"{label}: {pct:4.0f}%")
    return "  ".join(parts)


def print_results(r: dict, verbose: bool = True):
    sims = r["num_sims"]
    print(f"\n  ── sims = {sims:,} ──────────────────────────────────────────────")
    print(f"  Visited non-terminal nodes      : {r['total_visited']:>6,}")
    print(f"  Not-fully-expanded (NFE)        : {r['n_nfe']:>6,}  ({r['pct_nfe']:5.1f}% of visited)")
    print(f"    Pristine  (0 children tried)  : {r['n_pristine']:>6,}  ({r['pct_pristine_of_nfe']:5.1f}% of NFE)")
    print(f"    Partial   (≥1 child tried)    : {r['n_partial']:>6,}")

    print()
    print(f"  Expansion ratio  (n_tried / n_total)  for NFE nodes")
    print(f"    Unweighted:  mean={r['ratio_mean']:.3f}  p25={r['ratio_p25']:.3f}  p50={r['ratio_p50']:.3f}  p75={r['ratio_p75']:.3f}")
    print(f"    Visit-wtd :  mean={r['w_ratio_mean']:.3f}")

    print()
    print(f"  Pending batch size  (n_untried children)  for NFE nodes")
    print(f"    Unweighted:  mean={r['untr_mean']:.1f}  p25={r['untr_p25']:.0f}  p50={r['untr_p50']:.0f}  p75={r['untr_p75']:.0f}  max={r['untr_max']}")
    print(f"    Visit-wtd :  mean={r['w_n_untr_mean']:.1f}")

    print()
    print(f"  Total children (branching factor) for NFE nodes")
    print(f"    mean={r['tot_mean']:.1f}  p50={r['tot_p50']:.0f}   (visit-wtd mean={r['w_n_tot_mean']:.1f})")

    print()
    print(f"  Projected savings under 'expand all on first visit'")
    print(f"    Current NN calls        : {sims:>6,}  (1 per simulation)")
    print(f"    Calls under batch-all   : {r['total_visited']:>6,}  (1 per unique node, batched at branching-factor)")
    print(f"    Theoretical reduction   : {r['projected_savings_pct']:5.1f}%")

    if verbose and len(r["_n_untr"]) > 0:
        print()
        print("  Histogram — pending batch size (n_untried) for NFE nodes:")
        # Determine sensible bins
        max_untr = r["untr_max"]
        if max_untr <= 24:
            bins = list(range(0, max_untr + 2))
        else:
            bins = [0, 1, 2, 4, 8, 12, 16, 20, max_untr + 1]
        print("  " + _hist_str(r["_n_untr"], bins=bins, fmt=".0f"))

        print()
        print("  Histogram — expansion ratio r for NFE nodes:")
        print("  " + _hist_str(r["_ratios"], bins=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], fmt=".2f"))


def print_summary_table(all_results: list[dict]):
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    hdr = (f"{'sims':>6}  {'visited':>7}  {'%NFE':>5}  {'%prist':>6}  "
           f"{'r_wtd':>6}  {'untr_wtd':>8}  {'bf_wtd':>6}  {'saved%':>7}")
    print(hdr)
    print("-" * 80)
    for r in all_results:
        print(f"{r['num_sims']:>6,}  {r['total_visited']:>7,}  "
              f"{r['pct_nfe']:>5.1f}  {r['pct_pristine_of_nfe']:>6.1f}  "
              f"{r['w_ratio_mean']:>6.3f}  {r['w_n_untr_mean']:>8.1f}  "
              f"{r['w_n_tot_mean']:>6.1f}  {r['projected_savings_pct']:>7.1f}")
    print()
    print("  %NFE     = % of visited nodes still not fully expanded")
    print("  %prist   = % of NFE nodes that have never had any child tried (full batch)")
    print("  r_wtd    = visit-weighted mean expansion ratio (tried/total) for NFE nodes")
    print("  untr_wtd = visit-weighted mean untried children (= pending batch size)")
    print("  bf_wtd   = visit-weighted mean branching factor for NFE nodes")
    print("  saved%   = projected NN-call reduction under batch-all-on-first-expansion")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pa = argparse.ArgumentParser(
        description="Analyse MCTS expansion patterns to assess batching viability."
    )
    pa.add_argument("--sims",        type=int,   nargs="+", default=[128, 512, 2048],
                    help="MCTS simulation budgets to test (default: 128 512 2048)")
    pa.add_argument("--positions",   type=int,   default=10,
                    help="Random starting positions to average over per budget (default: 10)")
    pa.add_argument("--exploration", type=float, default=1.4)
    pa.add_argument("--checkpoint",  type=str,   default=None)
    pa.add_argument("--seed",        type=int,   default=7)
    pa.add_argument("--no-verbose",  action="store_true",
                    help="Suppress per-budget histograms, show summary table only")
    args = pa.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    ckpt = args.checkpoint or _find_checkpoint()
    model = _load_model(ckpt)

    print(f"\nPositions per budget: {args.positions}")
    print(f"Simulation budgets  : {args.sims}")

    # Pre-generate board positions so all budgets see the same starting states.
    boards = [CppBoardState.random_start(random.getrandbits(32))
              for _ in range(args.positions)]

    all_summaries = []
    for sims in args.sims:
        print(f"\n{'='*60}")
        print(f"Budget: {sims:,} simulations  ×  {args.positions} positions")
        print(f"{'='*60}")

        # Aggregate across all positions
        agg = dict(
            total_visited=[], n_nfe=[], pct_nfe=[], n_pristine=[], n_partial=[],
            pct_pristine_of_nfe=[],
            ratio_mean=[], ratio_p25=[], ratio_p50=[], ratio_p75=[],
            untr_mean=[], untr_p25=[], untr_p50=[], untr_p75=[], untr_max=[],
            tot_mean=[], tot_p50=[],
            w_ratio_mean=[], w_n_untr_mean=[], w_n_tot_mean=[],
            projected_savings_pct=[],
        )
        all_ratios  = []
        all_n_untr  = []
        all_n_tot   = []

        for pos_idx, board in enumerate(boards):
            tree = _run_mcts(board, model, sims, args.exploration)
            r    = analyse_tree(tree, sims)
            for key in agg:
                agg[key].append(r[key])
            all_ratios.extend(r["_ratios"].tolist())
            all_n_untr.extend(r["_n_untr"].tolist())
            all_n_tot.extend(r["_n_tot"].tolist())

        # Build mean result for this budget
        mean_r = {k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()}
        mean_r["num_sims"]   = sims
        mean_r["_ratios"]    = np.array(all_ratios)
        mean_r["_n_untr"]    = np.array(all_n_untr, dtype=np.int32)
        mean_r["_n_tot"]     = np.array(all_n_tot,  dtype=np.int32)
        mean_r["_pristine"]  = np.array([])  # not needed for pooled histograms
        mean_r["total_visited"]         = int(mean_r["total_visited"])
        mean_r["n_nfe"]                 = int(mean_r["n_nfe"])
        mean_r["n_pristine"]            = int(mean_r["n_pristine"])
        mean_r["n_partial"]             = int(mean_r["n_partial"])
        mean_r["untr_max"]              = int(mean_r["untr_max"])

        print(f"\n  (Averages over {args.positions} positions)")
        print_results(mean_r, verbose=not args.no_verbose)
        all_summaries.append(mean_r)

    print_summary_table(all_summaries)


if __name__ == "__main__":
    main()
