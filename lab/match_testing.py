#!/usr/bin/env python3
"""
Match testing program: ABSearchBot vs AdvancedSearchBot
Runs tournaments at different time controls and records win rates.
"""

import time
from pathlib import Path
from onitama.Boardstate import Boardstate
from onitama.opponents.ab_search_bot import ABSearchBot
from onitama.opponents.advanced_search_bot import AdvancedSearchBot


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

def play_game(red_bot, blue_bot, time_limit, board=None):
    """Play a single game between two bots. Returns 'RED' or 'BLUE' for winner."""
    if board is None:
        board = Boardstate()
    
    move_count = 0
    while True:
        # Get the bot for current turn
        if board.turn_colour() == "RED":
            move = red_bot.request_move(board, time_limit=time_limit)
        else:
            move = blue_bot.request_move(board, time_limit=time_limit)
        
        board.execute_move(move)
        move_count += 1
        
        # Check for win
        result = board.is_won()
        if result is not None and result[0]:
            return result[0]
        
        # Safety check for infinite loops
        if move_count > 1000:
            print(f"Warning: Game exceeded 1000 moves")
            break
    
    return None

def run_tournament(num_matches, time_limit):
    """Run a tournament of num_matches between the two bots."""
    ab_bot = ABSearchBot(max_depth=6)
    adv_bot = AdvancedSearchBot(max_depth=6)
    
    ab_wins = 0
    adv_wins = 0
    
    for i in range(num_matches):
        print(f"  Match {i+1:3d}/{num_matches} at {time_limit}s... ", end="", flush=True)
        
        # Alternate who goes first for fairness
        if i % 2 == 0:
            # ABSearchBot is RED, AdvancedSearchBot is BLUE
            winner = play_game(ab_bot, adv_bot, time_limit)
            if winner == "RED":
                ab_wins += 1
                print("AB wins")
            else:
                adv_wins += 1
                print("ADV wins")
        else:
            # AdvancedSearchBot is RED, ABSearchBot is BLUE
            winner = play_game(adv_bot, ab_bot, time_limit)
            if winner == "RED":
                adv_wins += 1
                print("ADV wins")
            else:
                ab_wins += 1
                print("AB wins")
    
    return ab_wins, adv_wins

def output(message, file_obj=None):
    """Print to stdout and optionally to a file."""
    print(message)
    if file_obj:
        file_obj.write(message + "\n")

if __name__ == "__main__":
    # Open results file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "tournament_results.txt"

    with open(results_path, "w") as results_file:
        output("=" * 70, results_file)
        output("Bot Tournament: ABSearchBot vs AdvancedSearchBot", results_file)
        output("=" * 70, results_file)
        
        start_time = time.time()
        
        output("\n--- Running 100 matches at 0.1 second time limit ---", results_file)
        ab_01s, adv_01s = run_tournament(100, 0.1)
        
        output(f"\nResults at 0.1s:", results_file)
        output(f"  ABSearchBot:       {ab_01s:3d} wins ({ab_01s:3.1f}%)", results_file)
        output(f"  AdvancedSearchBot: {adv_01s:3d} wins ({adv_01s:3.1f}%)", results_file)
        
        output("\n--- Running 100 matches at 1.0 second time limit ---", results_file)
        ab_1s, adv_1s = run_tournament(100, 1.0)
        
        output(f"\nResults at 1.0s:", results_file)
        output(f"  ABSearchBot:       {ab_1s:3d} wins ({ab_1s:3.1f}%)", results_file)
        output(f"  AdvancedSearchBot: {adv_1s:3d} wins ({adv_1s:3.1f}%)", results_file)
        
        output("\n--- Running 100 matches at 3.0 second time limit ---", results_file)
        ab_3s, adv_3s = run_tournament(100, 3.0)
        
        output(f"\nResults at 3.0s:", results_file)
        output(f"  ABSearchBot:       {ab_3s:3d} wins ({ab_3s:3.1f}%)", results_file)
        output(f"  AdvancedSearchBot: {adv_3s:3d} wins ({adv_3s:3.1f}%)", results_file)
        
        elapsed = time.time() - start_time
        
        output("\n" + "=" * 70, results_file)
        output("Overall Summary:", results_file)
        ab_total = ab_01s + ab_1s + ab_3s
        adv_total = adv_01s + adv_1s + adv_3s
        output(f"  ABSearchBot:       {ab_total:3d} / 300 wins ({ab_total/3:5.1f}%)", results_file)
        output(f"  AdvancedSearchBot: {adv_total:3d} / 300 wins ({adv_total/3:5.1f}%)", results_file)
        output(f"  Total time elapsed: {elapsed:.1f} seconds", results_file)
        output("=" * 70, results_file)
        
        print(f"\nResults saved to {results_path}")
