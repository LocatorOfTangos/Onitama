"""Compile/run benchmark harness for pmcts_bot.cpp.

This script does two things:
1) Checks for compile errors in pmcts_bot.cpp.
2) If compilation succeeds, runs performance benchmarks (nodes/rollouts per second)
   via a generated C++ harness.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from typing import List


@dataclass
class RunResult:
    threads: int
    time_ms: int
    elapsed_s: float
    nodes_created: int
    rollouts_evaluated: int
    move_card_idx: int


ROOT = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
RESULTS_DIR = ROOT / "results"
CPP_FILE = PROJECT_ROOT / "src" / "onitama" / "opponents" / "pmcts" / "pmcts_bot.cpp"
BUILD_DIR = RESULTS_DIR / "pmcts_build"


def run_cmd(cmd: List[str], cwd: pathlib.Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )


def compile_source_only(cxx: str, std: str) -> tuple[bool, str]:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    obj_path = BUILD_DIR / "pmcts_bot.o"
    proc = run_cmd([
        cxx,
        f"-std={std}",
        "-O3",
        "-pthread",
        "-c",
        str(CPP_FILE),
        "-o",
        str(obj_path),
    ])
    if proc.returncode != 0:
        return False, (proc.stdout + "\n" + proc.stderr).strip()
    return True, ""


def write_harness(path: pathlib.Path) -> None:
    cpp_abs = CPP_FILE.resolve().as_posix()
    harness = textwrap.dedent(
        f"""
        #include <chrono>
        #include <iomanip>
        #include <iostream>
        #include <memory>
        #include <string>

        #define main pmcts_bot_original_main
        #include \"{cpp_abs}\"
        #undef main

        int main(int argc, char** argv) {{
            if (argc < 3) {{
                std::cerr << "usage: harness <threads> <time_ms>\\n";
                return 2;
            }}

            int threads = std::stoi(argv[1]);
            int time_ms = std::stoi(argv[2]);

            try {{
                MCTSBot bot(threads, 20.0, time_ms);
                auto board = std::make_shared<BoardState>();

                auto start = std::chrono::steady_clock::now();
                Move move = bot.search(board);
                auto end = std::chrono::steady_clock::now();

                double elapsed = std::chrono::duration<double>(end - start).count();
                int nodes = bot.nodes_created.load();
                int rollouts = bot.rollouts_evaluated.load();

                std::cout << "elapsed_s=" << std::fixed << std::setprecision(6) << elapsed << "\\n";
                std::cout << "nodes_created=" << nodes << "\\n";
                std::cout << "rollouts_evaluated=" << rollouts << "\\n";
                std::cout << "move_card_idx=" << move.card_idx << "\\n";
                return 0;
            }} catch (const std::exception& ex) {{
                std::cerr << "runtime_error=" << ex.what() << "\\n";
                return 1;
            }} catch (...) {{
                std::cerr << "runtime_error=unknown\\n";
                return 1;
            }}
        }}
        """
    ).strip() + "\n"
    path.write_text(harness, encoding="utf-8")


def compile_harness(cxx: str, std: str, harness_cpp: pathlib.Path, out_bin: pathlib.Path) -> tuple[bool, str]:
    proc = run_cmd([
        cxx,
        f"-std={std}",
        "-O3",
        "-pthread",
        str(harness_cpp),
        "-o",
        str(out_bin),
    ], cwd=ROOT)
    if proc.returncode != 0:
        return False, (proc.stdout + "\n" + proc.stderr).strip()
    return True, ""


def parse_run_output(output: str, threads: int, time_ms: int) -> RunResult:
    def pick_int(name: str) -> int:
        m = re.search(rf"{name}=(-?\d+)", output)
        if not m:
            raise ValueError(f"missing {name} in output")
        return int(m.group(1))

    def pick_float(name: str) -> float:
        m = re.search(rf"{name}=(-?\d+(?:\.\d+)?)", output)
        if not m:
            raise ValueError(f"missing {name} in output")
        return float(m.group(1))

    return RunResult(
        threads=threads,
        time_ms=time_ms,
        elapsed_s=pick_float("elapsed_s"),
        nodes_created=pick_int("nodes_created"),
        rollouts_evaluated=pick_int("rollouts_evaluated"),
        move_card_idx=pick_int("move_card_idx"),
    )


def print_compiler_error(err: str) -> None:
    print("\nCompilation failed.")
    print("-" * 80)
    print(err)
    print("-" * 80)


def print_results(results: list[RunResult]) -> None:
    print("\nParallel MCTS C++ Performance Testing")
    print("=" * 80)
    for row in results:
        rps = row.rollouts_evaluated / row.elapsed_s if row.elapsed_s > 0 else 0.0
        nps = row.nodes_created / row.elapsed_s if row.elapsed_s > 0 else 0.0
        print(f"\nthreads={row.threads}, time_limit={row.time_ms}ms")
        print("-" * 80)
        print(f"  Elapsed time: {row.elapsed_s:.3f}s")
        print(f"  MCTS nodes created: {row.nodes_created}")
        print(f"  Rollouts evaluated: {row.rollouts_evaluated}")
        print(f"  Rollouts per second: {rps:.0f}")
        print(f"  Nodes per second: {nps:.0f}")
        print(f"  Returned move card idx: {row.move_card_idx}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark and error-check pmcts_bot.cpp")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"), help="C++ compiler executable")
    parser.add_argument("--std", default="c++17", help="C++ language standard")
    parser.add_argument("--threads", default="1,2,4", help="Comma-separated thread counts")
    parser.add_argument("--times", default="500,1000,2000", help="Comma-separated time limits in ms")
    args = parser.parse_args()

    if not CPP_FILE.exists():
        print(f"Error: {CPP_FILE} not found")
        return 2

    # First: compile source only to surface direct errors clearly.
    ok, err = compile_source_only(args.cxx, args.std)
    if not ok:
        print("Source compile check: FAILED")
        print_compiler_error(err)
        return 1
    print("Source compile check: OK")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    harness_cpp = BUILD_DIR / "parallel_mcts_benchmark_harness.cpp"
    harness_bin = BUILD_DIR / "parallel_mcts_benchmark_harness"
    write_harness(harness_cpp)

    ok, err = compile_harness(args.cxx, args.std, harness_cpp, harness_bin)
    if not ok:
        print("Harness compile check: FAILED")
        print_compiler_error(err)
        return 1
    print("Harness compile check: OK")

    thread_values = [int(x.strip()) for x in args.threads.split(",") if x.strip()]
    time_values = [int(x.strip()) for x in args.times.split(",") if x.strip()]

    results: list[RunResult] = []
    for t in thread_values:
        for ms in time_values:
            proc = run_cmd([str(harness_bin), str(t), str(ms)], cwd=ROOT)
            if proc.returncode != 0:
                print(f"\nRun failed (threads={t}, time_ms={ms})")
                print("-" * 80)
                print((proc.stdout + "\n" + proc.stderr).strip())
                print("-" * 80)
                return 1
            try:
                row = parse_run_output(proc.stdout, t, ms)
            except Exception as ex:
                print(f"\nOutput parse failed (threads={t}, time_ms={ms}): {ex}")
                print("-" * 80)
                print(proc.stdout)
                print("-" * 80)
                return 1
            results.append(row)

    print_results(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
