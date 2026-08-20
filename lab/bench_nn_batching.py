"""
bench_nn_batching.py — Measure ValueNet1 inference cost at varying batch sizes
on CPU and GPU to determine whether GPU batching would save time in self-play.

What it measures (for each device × batch-size combination):
  • Full round-trip latency: allocate → fill → transfer → forward → .item() sync
    This is the actual cost paid per MCTS "ply batch" in production.
  • Throughput: samples/second.

Then it computes, for your configured simulation budget and branching factor:
  • Current cost per ply  = sims × CPU_batch1_latency
  • GPU batch-N cost per ply = ceil(sims / N) × GPU_batchN_latency
  • Net saving (%) and absolute saving (ms) per ply

The "minimum plausible overhead" for moving to GPU batching is:
  GPU_batch1_latency  — the floor you pay even with no parallelism benefit,
                        just from the PCIe transfer and kernel-launch overhead.

Usage:
    python lab/bench_nn_batching.py
    python lab/bench_nn_batching.py --sims 2048 --branching 13 --warmup 50 --reps 300
    python lab/bench_nn_batching.py --no-gpu          # CPU-only comparison
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from onitama.opponents.NN.architecture.ValueNet1 import ValueNet1

BATCH_SIZES = [1, 2, 4, 8, 12, 16, 24, 32, 64]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: str | None, device: torch.device) -> torch.nn.Module:
    model = ValueNet1().to(device)
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    # JIT-trace on the target device, same as production workers
    try:
        dummy_b = torch.empty(1, 4, 5, 5, device=device)
        dummy_c = torch.empty(1, 3, 16,  device=device)
        model = torch.jit.trace(model, (dummy_b, dummy_c))
        model.eval()
    except Exception as e:
        print(f"  JIT trace failed ({e}); using eager.")
    return model


def _find_checkpoint() -> str | None:
    d = _SRC / "onitama" / "opponents" / "NN" / "logs3"
    for name in ("vn1_stage3_checkpoint.pt", "vn1_stage3_pb_checkpoint.pt"):
        p = d / name
        if p.exists():
            return str(p)
    return None


def _gpu_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# Benchmark: full round-trip latency for a given device and batch size
# ---------------------------------------------------------------------------

def _bench_one(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    warmup: int,
    reps: int,
    src_device: torch.device,   # where the input tensors start (always CPU)
) -> tuple[float, float]:
    """
    Returns (mean_latency_s, std_latency_s) for one forward pass of size batch_size.

    The measured window covers the realistic production pipeline:
      1. Unpack batch_size × (board, cards) tensors stored on src_device (CPU pinned).
      2. Transfer to `device` (no-op if device==CPU).
      3. model() forward.
      4. .item() on the result (forces GPU sync; equivalent to reading the value
         back as Python does in the MCTS loop).
    """
    # Pre-allocate inputs on CPU (pinned if GPU target, for fastest transfer)
    pin = (device.type == "cuda")
    boards = torch.randn(batch_size, 4, 5, 5, pin_memory=pin)
    cards  = torch.randn(batch_size, 3, 16,  pin_memory=pin)

    # Warm up
    with torch.no_grad():
        for _ in range(warmup):
            b = boards.to(device, non_blocking=False)
            c = cards.to(device,  non_blocking=False)
            out = model(b, c)
            _ = out.sum().item()   # sync
    _gpu_sync(device)

    # Timed reps
    latencies = []
    with torch.no_grad():
        for _ in range(reps):
            _gpu_sync(device)
            t0 = time.perf_counter()

            b   = boards.to(device, non_blocking=False)
            c   = cards.to(device,  non_blocking=False)
            out = model(b, c)
            _   = out.sum().item()   # forces GPU sync + Python scalar conversion

            _gpu_sync(device)
            latencies.append(time.perf_counter() - t0)

    mu  = sum(latencies) / len(latencies)
    std = math.sqrt(sum((x - mu) ** 2 for x in latencies) / len(latencies))
    return mu, std


# ---------------------------------------------------------------------------
# Analysis: projected per-ply cost under different batching strategies
# ---------------------------------------------------------------------------

def analyse(results: dict, sims: int, branching: int):
    """
    results: {(device_label, batch_size): (mean_s, std_s)}
    Prints projected per-ply wall-clock under each strategy.
    """
    cpu_b1 = results.get(("CPU", 1))
    if cpu_b1 is None:
        return
    baseline_ply_ms = sims * cpu_b1[0] * 1e3

    print()
    print("=" * 78)
    print("PROJECTED PER-PLY COST  (sims=%d, avg branching≈%d)" % (sims, branching))
    print("=" * 78)
    print(f"  Baseline (CPU, batch=1, sequential): {baseline_ply_ms:8.1f} ms/ply")
    print()
    print(f"  {'Strategy':<32}  {'calls/ply':>9}  {'ms/ply':>8}  {'saving':>8}  {'saving ms':>10}")
    print("  " + "-" * 72)

    strategies = []
    for (dev, bs), (mu, std) in sorted(results.items(), key=lambda x: (x[0][0], x[0][1])):
        # How many NN calls per ply under this batch size?
        # We assume ceil(sims / bs) batches of size bs are formed.
        # (Conservative: ignores that terminal nodes need no NN call,
        #  and that batches at the end of the budget may be smaller.)
        calls = math.ceil(sims / bs)
        ply_ms = calls * mu * 1e3
        saving_pct = 100.0 * (baseline_ply_ms - ply_ms) / baseline_ply_ms
        saving_ms  = baseline_ply_ms - ply_ms
        label = f"{dev} batch={bs}"
        strategies.append((label, calls, ply_ms, saving_pct, saving_ms))

    for label, calls, ply_ms, saving_pct, saving_ms in strategies:
        marker = " ◀ break-even" if abs(saving_pct) < 2 else (" ✓" if saving_pct > 5 else "")
        print(f"  {label:<32}  {calls:>9,}  {ply_ms:>8.1f}  {saving_pct:>7.1f}%  {saving_ms:>9.1f}{marker}")

    print()
    # Also print: at what batch size does CPU catch up to GPU?
    gpu_b1 = results.get(("GPU", 1))
    if gpu_b1 is not None:
        gpu_overhead_ms = gpu_b1[0] * 1e3
        print(f"  GPU single-sample overhead (batch=1): {gpu_overhead_ms:.2f} ms")
        print(f"  = {gpu_overhead_ms / (cpu_b1[0]*1e3):.1f}× CPU batch=1 latency")
        print()
        print("  Break-even: GPU batch-N beats CPU batch-1 when")
        print(f"    ceil({sims}/N) × GPU_batchN < {sims} × {cpu_b1[0]*1e3:.3f} ms")
        # Find smallest N where GPU wins
        for bs in BATCH_SIZES:
            key = ("GPU", bs)
            if key not in results:
                continue
            mu_gpu = results[key][0]
            calls_gpu = math.ceil(sims / bs)
            if calls_gpu * mu_gpu < sims * cpu_b1[0]:
                print(f"    → First GPU win at batch={bs}  "
                      f"({calls_gpu}×{mu_gpu*1e3:.2f}ms = {calls_gpu*mu_gpu*1e3:.1f}ms "
                      f"vs {sims*cpu_b1[0]*1e3:.1f}ms)")
                break
        else:
            print("    → No GPU batch size tested shows a saving over CPU baseline.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pa = argparse.ArgumentParser(
        description="Benchmark ValueNet1 inference cost vs batch size on CPU and GPU."
    )
    pa.add_argument("--sims",       type=int,   default=2048,
                    help="Simulations per ply in your training run (default: 2048)")
    pa.add_argument("--branching",  type=int,   default=13,
                    help="Average branching factor for context (default: 13)")
    pa.add_argument("--warmup",     type=int,   default=50,
                    help="Warmup forward passes before timing (default: 50)")
    pa.add_argument("--reps",       type=int,   default=200,
                    help="Timed repetitions per (device, batch) (default: 200)")
    pa.add_argument("--checkpoint", type=str,   default=None)
    pa.add_argument("--no-gpu",     action="store_true",
                    help="Skip GPU benchmarks (CPU only)")
    pa.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES,
                    help="Batch sizes to test")
    args = pa.parse_args()

    ckpt = args.checkpoint or _find_checkpoint()
    has_gpu = torch.cuda.is_available() and not args.no_gpu

    devices = [("CPU", torch.device("cpu"))]
    if has_gpu:
        devices.append(("GPU", torch.device("cuda")))
    else:
        if not args.no_gpu:
            print("NOTE: No CUDA device found; running CPU-only.")

    cpu_device = torch.device("cpu")
    results: dict[tuple[str, int], tuple[float, float]] = {}

    for dev_label, device in devices:
        print(f"\n{'='*60}")
        print(f"Device: {dev_label}  ({device})")
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            print(f"  {props.name}  {props.total_memory // 2**20} MiB")
        print(f"{'='*60}")

        model = _load_model(ckpt, device)

        print(f"  {'batch':>6}  {'mean (µs)':>11}  {'std (µs)':>10}  {'throughput (samp/s)':>20}")
        print("  " + "-" * 55)

        for bs in args.batch_sizes:
            mu, std = _bench_one(
                model=model,
                device=device,
                batch_size=bs,
                warmup=args.warmup,
                reps=args.reps,
                src_device=cpu_device,
            )
            throughput = bs / mu
            print(f"  {bs:>6}  {mu*1e6:>11.1f}  {std*1e6:>10.1f}  {throughput:>20,.0f}")
            results[(dev_label, bs)] = (mu, std)

    analyse(results, sims=args.sims, branching=args.branching)


if __name__ == "__main__":
    main()
