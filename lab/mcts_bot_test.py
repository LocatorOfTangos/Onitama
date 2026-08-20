"""Test/benchmark script for MCTS bot instrumentation."""
from onitama import Boardstate
import time
import sys
from onitama.opponents.mcts_bot import MCTSBot


print("MCTS Bot Performance Testing")
print("=" * 80)
print()

# Test with different time limits and parameters
for time_limit in [0.5, 1.0, 2.0]:
    print(f"\nTime limit: {time_limit}s")
    print("-" * 80)
    
    if len(sys.argv) > 1 and sys.argv[1] == "temp_none":
        print("Testing MCTSBot with efficient random rollouts")
        bot = MCTSBot(temperature=None)
    else:
        bot = MCTSBot()
    
    board = Boardstate.Boardstate()
    
    start_time = time.time()
    move = bot.request_move(board, time_limit=time_limit)
    elapsed = time.time() - start_time
    
    stats = bot.get_stats()
    
    rollouts_per_sec = stats['rollouts_evaluated'] / elapsed if elapsed > 0 else 0
    nodes_per_sec = stats['nodes_created'] / elapsed if elapsed > 0 else 0
    copies_per_sec = stats['board_copies'] / elapsed if elapsed > 0 else 0
    time_in_rollouts = stats.get('time_in_rollouts', 0.0)
    rollouts_per_sec_during = stats['rollouts_evaluated'] / time_in_rollouts if time_in_rollouts > 0 else 0
    
    print(f"  Elapsed time: {elapsed:.2f}s")
    print(f"  MCTS nodes created: {stats['nodes_created']}")
    print(f"  Rollouts evaluated: {stats['rollouts_evaluated']}")
    print(f"  Board copies: {stats['board_copies']}")
    print(f"  Time spent in rollouts: {time_in_rollouts:.2f}s")
    print(f"  Rollouts per second: {rollouts_per_sec:.0f} ({rollouts_per_sec_during:.0f} during rollouts)")
    print(f"  Nodes per second: {nodes_per_sec:.0f}")
    print(f"  Copies per second: {copies_per_sec:.0f}")
