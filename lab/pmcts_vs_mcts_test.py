#!/usr/bin/env python3
"""Head-to-head benchmark: Python MCTS vs C++ PMCTS.

Runs a color-balanced match set and reports:
- wins/losses/draws
- average move time per bot
- total think time per bot
"""

import argparse
import csv
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from onitama.Boardstate import Boardstate
from onitama.opponents.mcts_bot import MCTSBot
from onitama.opponents.pmcts.pmcts_bot import PMCTSBot


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


@dataclass
class BotStats:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    move_count: int = 0
    think_time_s: float = 0.0

    def avg_move_ms(self) -> float:
        return (self.think_time_s * 1000.0 / self.move_count) if self.move_count else 0.0


def play_game(red_bot, blue_bot, red_name: str, blue_name: str, time_limit: float, max_moves: int):
    board = Boardstate()
    move_count = 0
    timing = {red_name: [0.0, 0], blue_name: [0.0, 0]}  # think_time, moves

    while True:
        turn = board.turn_colour()
        if turn == "RED":
            bot = red_bot
            bot_name = red_name
        else:
            bot = blue_bot
            bot_name = blue_name

        start = time.perf_counter()
        move = bot.request_move(board, time_limit=time_limit)
        elapsed = time.perf_counter() - start

        timing[bot_name][0] += elapsed
        timing[bot_name][1] += 1

        if move is None:
            winner = "BLUE" if turn == "RED" else "RED"
            return winner, move_count, timing

        board.execute_move(move)
        move_count += 1

        result = board.is_won()
        if result is not None:
            return result[0], move_count, timing

        if move_count >= max_moves:
            return None, move_count, timing


def run_head_to_head(matches: int, time_limit: float, max_moves: int,
                    pmcts_threads: int, pmcts_temp: float, pmcts_explore: float,
                    mcts_temp: float | None, mcts_explore: float, seed: int | None):
    if seed is not None:
        random.seed(seed)

    mcts_stats = BotStats()
    pmcts_stats = BotStats()

    for match_idx in range(matches):
        mcts_bot = MCTSBot(temperature=mcts_temp, exploration=mcts_explore)
        pmcts_bot = PMCTSBot(
            num_threads=pmcts_threads,
            temperature=pmcts_temp,
            exploration=pmcts_explore,
        )

        pmcts_is_red = (match_idx % 2 == 0)
        if pmcts_is_red:
            winner, move_count, timing = play_game(pmcts_bot, mcts_bot, "PMCTS", "MCTS", time_limit, max_moves)
        else:
            winner, move_count, timing = play_game(mcts_bot, pmcts_bot, "MCTS", "PMCTS", time_limit, max_moves)

        pmcts_stats.think_time_s += timing["PMCTS"][0]
        pmcts_stats.move_count += timing["PMCTS"][1]
        mcts_stats.think_time_s += timing["MCTS"][0]
        mcts_stats.move_count += timing["MCTS"][1]

        if winner is None:
            pmcts_stats.draws += 1
            mcts_stats.draws += 1
            outcome = "DRAW"
        else:
            pmcts_won = (winner == "RED" and pmcts_is_red) or (winner == "BLUE" and not pmcts_is_red)
            if pmcts_won:
                pmcts_stats.wins += 1
                mcts_stats.losses += 1
                outcome = "PMCTS"
            else:
                pmcts_stats.losses += 1
                mcts_stats.wins += 1
                outcome = "MCTS"

        print(f"Match {match_idx + 1:3d}/{matches}: winner={outcome:5s} moves={move_count:3d} PMCTS_as={'RED' if pmcts_is_red else 'BLUE'}")

    return mcts_stats, pmcts_stats


def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def summarize_results(label: str, total_matches: int, mcts_stats: BotStats, pmcts_stats: BotStats, elapsed: float):
    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)
    print(f"PMCTS  wins={pmcts_stats.wins:4d} losses={pmcts_stats.losses:4d} draws={pmcts_stats.draws:4d}  winrate={pct(pmcts_stats.wins, total_matches):5.1f}%")
    print(f"MCTS   wins={mcts_stats.wins:4d} losses={mcts_stats.losses:4d} draws={mcts_stats.draws:4d}  winrate={pct(mcts_stats.wins, total_matches):5.1f}%")
    print("-")
    print(f"PMCTS  total_think={pmcts_stats.think_time_s:10.2f}s  moves={pmcts_stats.move_count:6d}  avg_move={pmcts_stats.avg_move_ms():7.1f} ms")
    print(f"MCTS   total_think={mcts_stats.think_time_s:10.2f}s  moves={mcts_stats.move_count:6d}  avg_move={mcts_stats.avg_move_ms():7.1f} ms")
    print(f"elapsed wall-clock: {elapsed:.2f}s")


def run_meaningful_batch(args, mcts_temp):
    rounds = args.batch_rounds
    matches_per_round = args.matches

    if args.seed is None:
        base_seed = random.randint(1, 10**9)
    else:
        base_seed = args.seed

    print("=" * 72)
    print("Head-to-head: PMCTS vs MCTS (Meaningful Batch)")
    print("=" * 72)
    print(f"rounds={rounds}, matches_per_round={matches_per_round}, total_matches={rounds * matches_per_round}")
    print(f"time_limit={args.time_limit}s, max_moves={args.max_moves}")
    print(f"PMCTS: threads={args.pmcts_threads}, temp={args.pmcts_temp}, explore={args.pmcts_explore}")
    print(f"MCTS : temp={mcts_temp}, explore={args.mcts_explore}")
    print(f"base_seed={base_seed}")
    print("-" * 72)

    total_mcts = BotStats()
    total_pmcts = BotStats()
    pmcts_round_winrates = []
    rows = []

    start_all = time.time()
    for round_idx in range(rounds):
        seed_i = base_seed + round_idx
        print(f"\n[Round {round_idx + 1}/{rounds}] seed={seed_i}")
        round_start = time.time()

        mcts_stats, pmcts_stats = run_head_to_head(
            matches=matches_per_round,
            time_limit=args.time_limit,
            max_moves=args.max_moves,
            pmcts_threads=args.pmcts_threads,
            pmcts_temp=args.pmcts_temp,
            pmcts_explore=args.pmcts_explore,
            mcts_temp=mcts_temp,
            mcts_explore=args.mcts_explore,
            seed=seed_i,
        )

        round_elapsed = time.time() - round_start
        pmcts_wr = pct(pmcts_stats.wins, matches_per_round)
        pmcts_round_winrates.append(pmcts_wr)

        total_mcts.wins += mcts_stats.wins
        total_mcts.losses += mcts_stats.losses
        total_mcts.draws += mcts_stats.draws
        total_mcts.move_count += mcts_stats.move_count
        total_mcts.think_time_s += mcts_stats.think_time_s

        total_pmcts.wins += pmcts_stats.wins
        total_pmcts.losses += pmcts_stats.losses
        total_pmcts.draws += pmcts_stats.draws
        total_pmcts.move_count += pmcts_stats.move_count
        total_pmcts.think_time_s += pmcts_stats.think_time_s

        print(f"Round summary: PMCTS {pmcts_stats.wins}-{pmcts_stats.losses}-{pmcts_stats.draws} ({pmcts_wr:.1f}%) | elapsed={round_elapsed:.1f}s")

        rows.append({
            "round": round_idx + 1,
            "seed": seed_i,
            "matches": matches_per_round,
            "pmcts_wins": pmcts_stats.wins,
            "pmcts_losses": pmcts_stats.losses,
            "pmcts_draws": pmcts_stats.draws,
            "pmcts_winrate_pct": pmcts_wr,
            "mcts_wins": mcts_stats.wins,
            "mcts_losses": mcts_stats.losses,
            "mcts_draws": mcts_stats.draws,
            "pmcts_avg_move_ms": pmcts_stats.avg_move_ms(),
            "mcts_avg_move_ms": mcts_stats.avg_move_ms(),
            "pmcts_total_think_s": pmcts_stats.think_time_s,
            "mcts_total_think_s": mcts_stats.think_time_s,
            "round_elapsed_s": round_elapsed,
        })

    total_matches = rounds * matches_per_round
    elapsed_all = time.time() - start_all
    summarize_results("Aggregate Results", total_matches, total_mcts, total_pmcts, elapsed_all)

    mean_wr = statistics.mean(pmcts_round_winrates) if pmcts_round_winrates else 0.0
    if len(pmcts_round_winrates) >= 2:
        stdev_wr = statistics.stdev(pmcts_round_winrates)
        ci95 = 1.96 * stdev_wr / (len(pmcts_round_winrates) ** 0.5)
    else:
        ci95 = 0.0

    print(f"Round winrate mean (PMCTS): {mean_wr:.2f}%")
    print(f"Round winrate 95% CI approx: ±{ci95:.2f}%")

    if args.output_csv:
        output_csv_path = Path(args.output_csv)
        if not output_csv_path.is_absolute():
            output_csv_path = RESULTS_DIR / output_csv_path
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        print(f"Saved round-by-round CSV to: {output_csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run PMCTS vs MCTS benchmark")
    parser.add_argument("--matches", type=int, default=40, help="Number of games to play (or per round in batch mode)")
    parser.add_argument("--time-limit", type=float, default=1.0, help="Seconds per move")
    parser.add_argument("--max-moves", type=int, default=400, help="Declare draw after this many moves")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--meaningful-batch", action="store_true",
                        help="Run multiple seeded rounds and aggregate stats")
    parser.add_argument("--batch-rounds", type=int, default=6,
                        help="Number of rounds for meaningful batch mode")
    parser.add_argument("--output-csv", type=str, default=str(RESULTS_DIR / "pmcts_vs_mcts_batch.csv"),
                        help="CSV output path for batch mode")

    parser.add_argument("--pmcts-threads", type=int, default=8, help="PMCTS worker threads")
    parser.add_argument("--pmcts-temp", type=float, default=0.7, help="PMCTS rollout temperature (<=0 => random)")
    parser.add_argument("--pmcts-explore", type=float, default=20.0, help="PMCTS UCB exploration constant")

    parser.add_argument("--mcts-temp", type=float, default=0.7, help="MCTS rollout temperature")
    parser.add_argument("--mcts-random-rollout", action="store_true", help="Use random rollouts for MCTS (temperature=None)")
    parser.add_argument("--mcts-explore", type=float, default=20.0, help="MCTS UCB exploration constant")

    args = parser.parse_args()
    mcts_temp = None if args.mcts_random_rollout else args.mcts_temp

    if args.meaningful_batch:
        run_meaningful_batch(args, mcts_temp)
        return

    print("=" * 72)
    print("Head-to-head: PMCTS vs MCTS")
    print("=" * 72)
    print(f"matches={args.matches}, time_limit={args.time_limit}s, max_moves={args.max_moves}")
    print(f"PMCTS: threads={args.pmcts_threads}, temp={args.pmcts_temp}, explore={args.pmcts_explore}")
    print(f"MCTS : temp={mcts_temp}, explore={args.mcts_explore}")
    if args.seed is not None:
        print(f"seed={args.seed}")
    print("-" * 72)

    start = time.time()
    mcts_stats, pmcts_stats = run_head_to_head(
        matches=args.matches,
        time_limit=args.time_limit,
        max_moves=args.max_moves,
        pmcts_threads=args.pmcts_threads,
        pmcts_temp=args.pmcts_temp,
        pmcts_explore=args.pmcts_explore,
        mcts_temp=mcts_temp,
        mcts_explore=args.mcts_explore,
        seed=args.seed,
    )
    elapsed = time.time() - start

    total = args.matches
    summarize_results("Results", total, mcts_stats, pmcts_stats, elapsed)


if __name__ == "__main__":
    main()
