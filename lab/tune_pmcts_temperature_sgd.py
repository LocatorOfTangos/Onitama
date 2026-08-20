#!/usr/bin/env python3
"""
Tune PMCTS temperature with ADAM updates.

This script keeps PMCTS exploration fixed (default: 700.0) and updates only
`temperature` using finite-difference gradient estimates from head-to-head
matches, inspired by `tune_mcts_parameters.py`.
"""

import argparse
import json
import math
from pathlib import Path

from onitama import Boardstate
from onitama.opponents import PMCTSBot


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def play_game(bot_red, bot_blue, max_moves=65, time_limit=1.0):
    """Play one game and return (winner, moves_played).

    winner: "RED", "BLUE", or None for draw/timeout.
    """
    board = Boardstate.Boardstate()
    moves_played = 0

    while board.is_won() is None and moves_played < max_moves:
        if board.turn_colour() == "RED":
            move = bot_red.request_move(board, time_limit=time_limit)
        else:
            move = bot_blue.request_move(board, time_limit=time_limit)

        if move is None:
            return None, moves_played

        board.apply_move(move)
        moves_played += 1

    result = board.is_won()
    if result is None:
        return None, moves_played
    return result[0], moves_played


def evaluate_pair(temp_a, temp_b, exploration, num_games=10, time_limit=1.0, max_moves=65):
    """Evaluate temp_a against temp_b with color balancing.

    Returns:
        win_rate_a: expected score in [0, 1] for temp_a
        avg_moves: average game length
    """
    bot_a = PMCTSBot(
        time_limit_ms=int(time_limit * 1000),
        num_threads=8,
        temperature=temp_a,
        exploration=exploration,
    )
    bot_b = PMCTSBot(
        time_limit_ms=int(time_limit * 1000),
        num_threads=8,
        temperature=temp_b,
        exploration=exploration,
    )

    score_a = 0.0
    total_moves = 0

    for game_index in range(num_games):
        a_is_red = (game_index % 2 == 0)
        if a_is_red:
            winner, moves = play_game(bot_a, bot_b, max_moves=max_moves, time_limit=time_limit)
            if winner is None:
                score_a += 0.5
            elif winner == "RED":
                score_a += 1.0
        else:
            winner, moves = play_game(bot_b, bot_a, max_moves=max_moves, time_limit=time_limit)
            if winner is None:
                score_a += 0.5
            elif winner == "BLUE":
                score_a += 1.0

        total_moves += moves

    return score_a / num_games, total_moves / num_games


def clip_temperature(temp, min_temp=0.05, max_temp=5.0):
    return max(min_temp, min(max_temp, temp))


def save_history(history, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = []
    for item in history:
        payload.append(
            {
                "iteration": item["iteration"],
                "temperature": item["temperature"],
                "gradient_log_temp": item["gradient_log_temp"],
                "momentum_m": item["momentum_m"],
                "momentum_v": item["momentum_v"],
                "momentum_m_hat": item["momentum_m_hat"],
                "momentum_v_hat": item["momentum_v_hat"],
                "win_rate_plus_vs_minus": item["win_rate_plus_vs_minus"],
                "avg_moves": item["avg_moves"],
                "temp_plus": item["temp_plus"],
                "temp_minus": item["temp_minus"],
            }
        )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def tune_temperature_sgd(
    initial_temp=0.7,
    exploration=700.0,
    learning_rate=0.4,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    iterations=40,
    batch_size=8,
    time_limit=1.0,
    temp_factor=1.2,
    min_temp=0.05,
    max_temp=5.0,
    output_path=str(RESULTS_DIR / "pmcts_temperature_history.json"),
):
    """Run ADAM-style tuning for PMCTS temperature only."""
    if temp_factor <= 1.0:
        raise ValueError("temp_factor must be > 1.0")
    if not (0.0 < beta1 < 1.0):
        raise ValueError("beta1 must be in (0, 1)")
    if not (0.0 < beta2 < 1.0):
        raise ValueError("beta2 must be in (0, 1)")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0")

    temperature = clip_temperature(initial_temp, min_temp=min_temp, max_temp=max_temp)
    momentum_m = 0.0
    momentum_v = 0.0

    history = []

    print("PMCTS Temperature Tuning (ADAM)")
    print("=" * 80)
    print(f"Initial temperature: {temperature:.4f}")
    print(f"Exploration (fixed): {exploration:.1f}")
    print(f"Learning rate: {learning_rate}")
    print("Learning rate schedule: exponential decay, half-life=100 iterations")
    print(f"ADAM betas: beta1={beta1}, beta2={beta2}, epsilon={epsilon}")
    print(f"Iterations: {iterations}, batch_size: {batch_size}")
    print(f"Perturbation factor (base): x{temp_factor:.3f} / ÷{temp_factor:.3f}")
    print("Perturbation schedule: decay toward 1.0, half-life=100 iterations")
    print("=" * 80)
    print()

    for iteration in range(1, iterations + 1):
        decay = 0.5 ** ((iteration - 1) / 100.0)
        effective_temp_factor = 1.0 + (temp_factor - 1.0) * decay
        log_factor = math.log(effective_temp_factor)

        temp_plus = clip_temperature(
            temperature * effective_temp_factor,
            min_temp=min_temp,
            max_temp=max_temp,
        )
        temp_minus = clip_temperature(
            temperature / effective_temp_factor,
            min_temp=min_temp,
            max_temp=max_temp,
        )

        print(f"Iteration {iteration}/{iterations}")
        print(f"  effective_temp_factor: x{effective_temp_factor:.4f} / ÷{effective_temp_factor:.4f}")
        print(f"  Comparing temp+={temp_plus:.4f} vs temp-={temp_minus:.4f} ...", flush=True)

        win_rate_plus, avg_moves = evaluate_pair(
            temp_plus,
            temp_minus,
            exploration=exploration,
            num_games=batch_size,
            time_limit=time_limit,
        )

        gradient_log_temp = (win_rate_plus - 0.5) / log_factor

        # ADAM ascent on expected score in log-temperature space
        momentum_m = beta1 * momentum_m + (1.0 - beta1) * gradient_log_temp
        momentum_v = beta2 * momentum_v + (1.0 - beta2) * (gradient_log_temp ** 2)
        momentum_m_hat = momentum_m / (1.0 - (beta1 ** iteration))
        momentum_v_hat = momentum_v / (1.0 - (beta2 ** iteration))

        log_temp = math.log(max(temperature, 1e-12))
        effective_learning_rate = learning_rate * (0.5 ** ((iteration - 1) / 100.0))
        adam_step = effective_learning_rate * momentum_m_hat / (math.sqrt(momentum_v_hat) + epsilon)
        new_log_temp = log_temp + adam_step
        new_temp = clip_temperature(math.exp(new_log_temp), min_temp=min_temp, max_temp=max_temp)

        print(f"  win_rate(temp+ over temp-): {win_rate_plus:.3f}")
        print(f"  grad_log_temp: {gradient_log_temp:.4f}")
        print(f"  momentum_m: {momentum_m:.6f} (m_hat={momentum_m_hat:.6f})")
        print(f"  momentum_v: {momentum_v:.6f} (v_hat={momentum_v_hat:.6f})")
        print(f"  effective_lr: {effective_learning_rate:.6f}")
        print(f"  adam_step(log_temp): {adam_step:.6f}")
        print(f"  update: {temperature:.4f} -> {new_temp:.4f}")
        print(f"  avg_moves: {avg_moves:.1f}")
        print()

        history.append(
            {
                "iteration": iteration,
                "temperature": new_temp,
                "gradient_log_temp": gradient_log_temp,
                "momentum_m": momentum_m,
                "momentum_v": momentum_v,
                "momentum_m_hat": momentum_m_hat,
                "momentum_v_hat": momentum_v_hat,
                "win_rate_plus_vs_minus": win_rate_plus,
                "avg_moves": avg_moves,
                "temp_plus": temp_plus,
                "temp_minus": temp_minus,
            }
        )

        temperature = new_temp
        save_history(history, output_path)

    print("=" * 80)
    print(f"Recommended temperature: {temperature:.4f}")
    print(f"History saved to: {output_path}")
    return temperature, history


def parse_args():
    parser = argparse.ArgumentParser(description="Tune PMCTS temperature with ADAM while keeping exploration fixed")
    parser.add_argument("--initial-temp", type=float, default=0.7)
    parser.add_argument("--exploration", type=float, default=700.0)
    parser.add_argument("--lr", type=float, default=0.4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=1.0, help="Per-move time limit in seconds")
    parser.add_argument("--temp-factor", type=float, default=1.4)
    parser.add_argument("--min-temp", type=float, default=0.05)
    parser.add_argument("--max-temp", type=float, default=500.0)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "pmcts_temperature_history.json")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = RESULTS_DIR / output_path

    tune_temperature_sgd(
        initial_temp=args.initial_temp,
        exploration=args.exploration,
        learning_rate=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        iterations=args.iterations,
        batch_size=args.batch_size,
        time_limit=args.time_limit,
        temp_factor=args.temp_factor,
        min_temp=args.min_temp,
        max_temp=args.max_temp,
        output_path=str(output_path),
    )


if __name__ == "__main__":
    main()
