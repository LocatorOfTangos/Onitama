#!/usr/bin/env python3
"""
MCTS parameter tuning via stochastic gradient descent.

Computes numerical gradients by playing the current bot against perturbed versions,
then updates parameters in the direction of improvement.
"""

import sys
import time
import math
import json
from pathlib import Path
from onitama.opponents import mcts_bot
from onitama import Boardstate


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def play_game(bot1, bot2, max_moves=65):
    """Play a single game between two bots.
    
    Returns: (winner, num_moves) where winner is 0 (bot1 RED), 1 (bot2 BLUE), or None (draw/timeout)
    """
    board = Boardstate.Boardstate()
    moves_played = 0
    
    while board.is_won() is None and moves_played < max_moves:
        if board.turn_colour() == "RED":
            move = bot1.request_move(board, time_limit=1.0)
        else:
            move = bot2.request_move(board, time_limit=1.0)
        
        if move is None:
            return None, moves_played
        
        board.apply_move(move)
        moves_played += 1
    
    result = board.is_won()
    if result is None:
        return None, moves_played
    
    # RED wins -> bot1 wins; BLUE wins -> bot2 wins
    winner = 0 if result[0] == "RED" else 1
    return winner, moves_played


def play_match(bot_center, bot_perturbed, bot_center_as_red=True, num_games=10):
    """Play num_games between center and perturbed bot.
    
    Returns: win_rate_center (float in [0, 1]), avg_moves
    """
    center_wins = 0
    total_moves = 0
    
    for _ in range(num_games):
        if bot_center_as_red:
            winner, moves = play_game(bot_center, bot_perturbed)
        else:
            winner, moves = play_game(bot_perturbed, bot_center)
        
        if winner is None:
            center_wins += 0.5  # draw counts as 0.5 win
        elif (winner == 0) == bot_center_as_red:  # center is RED or BLUE?
            center_wins += 1
        
        total_moves += moves
    
    win_rate = center_wins / num_games
    avg_moves = total_moves / num_games
    return win_rate, avg_moves


def compute_gradient(temp, explore, temp_factor=1.2, explore_delta=1.0, num_games=5, time_limit=1.0):
    """Compute numerical gradient at (temp, explore).
    
    Tests current params against:
    - (temp * temp_factor, explore) and (temp / temp_factor, explore)  [log scale]
    - (temp, explore + explore_delta) and (temp, explore - explore_delta)  [linear scale]

    Returns: (grad_temp, grad_explore)
    where positive gradient means that direction improved win rate
    """
    center_bot = mcts_bot.MCTSBot(time_limit=time_limit, temperature=temp, exploration=explore)
    
    print(f"  Computing gradient at temp={temp:.3f}, explore={explore:.1f}")
    
    # Gradient in temperature direction (log scale: multiply)
    temp_higher = temp * temp_factor
    print(f"    Testing temp_higher={temp_higher:.3f}...", flush=True)
    bot_temp_plus = mcts_bot.MCTSBot(time_limit=time_limit, temperature=temp_higher, exploration=explore)
    wr_temp, _ = play_match(center_bot, bot_temp_plus, bot_center_as_red=True, num_games=num_games)
    
    # Gradient in exploration direction (linear scale: add)
    print(f"    Testing explore_higher={explore + explore_delta:.1f}...", flush=True)
    bot_explore_plus = mcts_bot.MCTSBot(time_limit=time_limit, temperature=temp, exploration=explore + explore_delta)
    wr_explore, _ = play_match(center_bot, bot_explore_plus, bot_center_as_red=True, num_games=num_games)
    
    # Numerical gradient: positive means center bot won more (so that direction is bad for us)
    # Negative gradient means we should move in that direction
    grad_temp = 0.5 - wr_temp  # If wr_temp > 0.5, perturbed won more, gradient is negative
    grad_explore = 0.5 - wr_explore
    
    print(f"    Results: temp_wr={wr_temp:.2f}, explore_wr={wr_explore:.2f}")
    print(f"    Gradient: temp_grad={grad_temp:.3f}, explore_grad={grad_explore:.3f}")
    
    return grad_temp, grad_explore


def clip_parameters(temp, explore):
    """Clip parameters to valid ranges."""
    temp = max(0.1, min(1000.0, temp))
    explore = max(1.0, min(50.0, explore))
    return temp, explore


def save_history(history, filename="param_history.json"):
    """Save parameter history to a JSON file."""
    output_path = Path(filename)
    if not output_path.is_absolute():
        output_path = RESULTS_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [{"iteration": i, "temperature": t, "exploration": e} for i, (t, e) in enumerate(history)]
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved history to {output_path}")


def adam_tune_mcts(initial_temp=1.0, initial_explore=10.0, 
                   learning_rate=0.1, num_iterations=20, batch_size=5,
                   beta1=0.9, beta2=0.999, epsilon=1e-8, time_limit=1.0):
    """Run ADAM to tune MCTS parameters.
    
    Args:
        initial_temp: Starting temperature
        initial_explore: Starting exploration coefficient
        learning_rate: ADAM learning rate
        num_iterations: Number of ADAM steps to take
        batch_size: Number of games per gradient estimate
        beta1: Exponential decay rate for first moment estimates
        beta2: Exponential decay rate for second moment estimates
        epsilon: Small constant for numerical stability
    """
    
    temp = initial_temp
    explore = initial_explore
    
    # ADAM state variables
    m_temp, m_explore = 0.0, 0.0  # First moments
    v_temp, v_explore = 0.0, 0.0  # Second moments
    
    print("MCTS Parameter Tuning via ADAM Optimization")
    print("=" * 80)
    print(f"Initial params: temp={temp:.3f}, explore={explore:.1f}")
    print(f"Learning rate: {learning_rate}")
    print(f"ADAM: beta1={beta1}, beta2={beta2}, epsilon={epsilon}")
    print(f"Iterations: {num_iterations}, batch_size per gradient: {batch_size}")
    print("=" * 80)
    print()
    
    history = [(temp, explore)]
    
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")
        
        # Compute gradient with adaptive perturbation magnitude
        temp_factor = 1.2  # Multiplicative factor for log scale (20% increase/decrease)
        explore_delta = max(0.5, explore * 0.1)  # 10% of current value, min 0.5
        
        grad_temp, grad_explore = compute_gradient(
            temp, explore, 
            temp_factor=temp_factor, 
            explore_delta=explore_delta,
            num_games=batch_size,
            time_limit=time_limit
        )
        
        # Normalize gradients to unit magnitude if both are non-zero
        grad_magnitude = math.sqrt(grad_temp ** 2 + grad_explore ** 2)
        if grad_magnitude > 1e-6:
            grad_temp /= grad_magnitude
            grad_explore /= grad_magnitude
        
        # ADAM update rule
        # Update biased first moment estimate
        m_temp = beta1 * m_temp + (1 - beta1) * grad_temp
        m_explore = beta1 * m_explore + (1 - beta1) * grad_explore
        
        # Update biased second raw moment estimate
        v_temp = beta2 * v_temp + (1 - beta2) * (grad_temp ** 2)
        v_explore = beta2 * v_explore + (1 - beta2) * (grad_explore ** 2)
        
        # Compute bias-corrected first moment estimate
        m_temp_hat = m_temp / (1 - beta1 ** (iteration + 1))
        m_explore_hat = m_explore / (1 - beta1 ** (iteration + 1))
        
        # Compute bias-corrected second raw moment estimate
        v_temp_hat = v_temp / (1 - beta2 ** (iteration + 1))
        v_explore_hat = v_explore / (1 - beta2 ** (iteration + 1))
        
        # Update parameters (move in direction of negative gradient)
        # For temperature (log scale): multiply by factor
        # For exploration (linear scale): add delta
        temp_log_delta = math.log(temp_factor)  # Convert multiplicative to additive in log space
        temp_step = learning_rate * m_temp_hat / (math.sqrt(v_temp_hat) + epsilon)
        explore_step = learning_rate * m_explore_hat / (math.sqrt(v_explore_hat) + epsilon)
        
        temp_new = temp * math.exp(-temp_step * temp_log_delta)
        explore_new = explore - explore_step * explore_delta
        
        # Clip to valid ranges
        temp_new, explore_new = clip_parameters(temp_new, explore_new)
        
        print(f"  Update: temp {temp:.6f} -> {temp_new:.6f}, explore {explore:.6f} -> {explore_new:.6f}")
        print(f"    grad_temp={grad_temp:.6f}, grad_explore={grad_explore:.6f}")
        print(f"    m_temp_hat={m_temp_hat:.6f}, m_explore_hat={m_explore_hat:.6f}")
        print(f"    v_temp_hat={v_temp_hat:.6f}, v_explore_hat={v_explore_hat:.6f}")
        print(f"    temp_step={temp_step:.6f}, explore_step={explore_step:.6f}")
        print(f"    temp_delta={temp_log_delta:.6f}, explore_delta={explore_delta:.6f}")
        print()
        
        temp = temp_new
        explore = explore_new
        history.append((temp, explore))
        
        # Save history after each update
        save_history(history)
    
    # Final evaluation against the initial parameters
    print("=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    print(f"Final parameters: temp={temp:.3f}, explore={explore:.1f}")
    print()
    
    print("Testing final bot vs initial bot (best of 3):")
    final_bot = mcts_bot.MCTSBot(time_limit=1.0, temperature=temp, exploration=explore)
    baseline_bot = mcts_bot.MCTSBot(time_limit=1.0, temperature=initial_temp, exploration=initial_explore)
    
    final_wr, avg_moves = play_match(final_bot, baseline_bot, bot_center_as_red=True, num_games=10)
    print(f"  Final bot win rate vs Initial bot: {final_wr:.2f}, avg_moves={avg_moves:.1f}")
    print()
    
    print("Parameter history:")
    for i, (t, e) in enumerate(history):
        print(f"  Step {i:2d}: temp={t:.3f}, explore={e:.1f}")
    
    return temp, explore, history


if __name__ == "__main__":
    # Run ADAM optimization
    final_temp, final_explore, history = adam_tune_mcts(
        initial_temp=0.7,
        initial_explore=20.0,
        learning_rate=0.2,
        num_iterations=1000,
        batch_size=5,
        time_limit=1.0
    )
    
    print()
    print("=" * 80)
    print("Optimization complete!")
    print(f"Recommended parameters: temp={final_temp:.3f}, explore={final_explore:.1f}")
