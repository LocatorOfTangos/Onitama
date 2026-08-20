"""
MCTS Bot for Onitama

Uses trained neural network weights to play via Monte Carlo Tree Search.
Automatically loads the latest checkpoint from MCTS_data directory.
Supports time-limited search with iterative deepening via simulation count.
"""

import os
import time
import pickle
import numpy as np
import random
from pathlib import Path

from MCTS import NeuralNet, MCTS, MCTSNode
from LeanBoardstate import Boardstate


class MCTSSearchTimeout(Exception):
    """Exception raised when search time limit is exceeded."""
    pass


class TimeConstrainedMCTS(MCTS):
    """MCTS with deadline-aware search."""
    
    def __init__(self, net, c_puct=1.0, n_simulations=800):
        super().__init__(net, c_puct=c_puct, n_simulations=n_simulations)
        self.deadline = None
        self.check_interval = 100  # Check deadline every N simulations
    
    def search_with_deadline(self, root_state, deadline):
        """Run MCTS simulations until deadline is reached.
        
        Args:
            root_state: Starting board state
            deadline: Absolute time (seconds) when search must complete
            
        Returns:
            Policy dict mapping move tuples to visit count probabilities
        """
        self.deadline = deadline
        root = MCTSNode(root_state)
        simulation_count = 0
        
        try:
            while time.time() < deadline:
                # Run a batch of simulations until next deadline check
                for _ in range(self.check_interval):
                    if time.time() >= deadline:
                        raise MCTSSearchTimeout()
                    
                    node = root
                    search_path = [node]

                    # Selection: traverse tree using UCB
                    while node.children and not self._is_terminal(node.state):
                        move_tuple, node = self._select_child(node)
                        search_path.append(node)

                    # Expansion: if not terminal, expand node with network priors
                    if not self._is_terminal(node.state):
                        policy, value = self._predict(node.state)
                        self._expand(node, policy)
                        # After expansion, select one of the new children to continue
                        if node.children:
                            move_tuple, node = self._select_child(node)
                            search_path.append(node)
                    else:
                        # Terminal node; evaluate outcome
                        value = self._terminal_value(node.state)

                    # Backup: propagate value back up the search path
                    self._backup(search_path, value)
                    simulation_count += 1

        except MCTSSearchTimeout:
            pass
        
        # Build policy from visit counts of root's children
        total_visits = sum(child.visit_count for child in root.children.values())
        pi = {}
        if total_visits > 0:
            for move_tuple, child in root.children.items():
                pi[move_tuple] = child.visit_count / total_visits
        
        return pi, simulation_count


class MCTSBot:
    """MCTS-based player using trained neural network with time limits."""
    
    def __init__(self, mcts_base_simulations=100, data_dir="MCTS_data", verbose=False):
        """Initialize MCTS bot with trained weights.
        
        Args:
            mcts_base_simulations: Base number of MCTS simulations (used without time limit)
            data_dir: Directory containing trained checkpoints
            verbose: Print debug information
        """
        self.mcts_base_simulations = mcts_base_simulations
        self.data_dir = data_dir
        self.verbose = verbose
        
        # Initialize network
        self.net = NeuralNet(seed=42)
        self.mcts = TimeConstrainedMCTS(self.net, n_simulations=mcts_base_simulations)
        
        # Try to load latest checkpoint
        self.checkpoint_loaded = self.load_latest_checkpoint()
        
        if not self.checkpoint_loaded:
            if verbose:
                print("Warning: No checkpoint found. Using random initialization.")
            self.checkpoint_path = None
        elif verbose:
            print(f"Loaded checkpoint: {self.checkpoint_path}")
    
    def load_latest_checkpoint(self):
        """Load the most recent checkpoint from data directory.
        
        Returns:
            True if checkpoint loaded successfully, False otherwise
        """
        latest_path = os.path.join(self.data_dir, "latest_checkpoint.pkl")
        
        if not os.path.exists(latest_path):
            return False
        
        try:
            with open(latest_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Restore network weights
            self.net.W = checkpoint['weights_W']
            self.net.b = checkpoint['weights_b']
            self.net.W_out = checkpoint['weights_W_out']
            self.net.b_out = checkpoint['weights_b_out']
            
            self.checkpoint_path = latest_path
            self.checkpoint_iteration = checkpoint.get('iteration', 0)
            
            if self.verbose:
                print(f"MCTS Bot: Loaded checkpoint from iteration {self.checkpoint_iteration}")
            
            return True
        except Exception as e:
            if self.verbose:
                print(f"Error loading checkpoint: {e}")
            return False
    
    def request_move(self, board, time_limit=None):
        """Select best move using MCTS with optional time limit.
        
        Args:
            board: Game board (Boardstate)
            time_limit: Maximum time in seconds. If None, uses base_simulations.
                       
        Returns:
            Selected move as (start_coord, end_coord, card_idx)
        """
        # Reload checkpoint to ensure we have latest weights
        self.load_latest_checkpoint()
        
        # Convert to LeanBoardstate if needed
        if isinstance(board, Boardstate):
            lean_state = board
        else:
            lean_state = board
        
        # Get legal moves for fallback
        legal_moves = lean_state.possible_moves()
        if not legal_moves:
            return None
        
        # Run MCTS search
        if time_limit is None:
            # Search with fixed number of simulations
            policy = self.mcts.search(lean_state)
        else:
            # Search with time limit
            deadline = time.time() + time_limit
            policy, sim_count = self.mcts.search_with_deadline(lean_state, deadline)
            if self.verbose:
                elapsed = time.time() - (deadline - time_limit)
                print(f"MCTS Bot: {sim_count} simulations in {elapsed:.2f}s")
        
        if not policy:
            # No legal moves in policy - use fallback
            move = random.choice(legal_moves)
            if self.verbose:
                print("MCTS Bot: No policy, using random move")
            return move
        
        # Select best move (highest visit count)
        best_move_idx = max(policy.keys(), key=lambda m: policy[m])
        start_idx, end_idx, card_idx = best_move_idx
        
        # Convert indices back to coordinates
        start_x, start_y = start_idx % 5, start_idx // 5
        end_x, end_y = (end_idx - 25) % 5, (end_idx - 25) // 5
        
        move = ((start_x, start_y), (end_x, end_y), card_idx)
        
        if self.verbose:
            confidence = policy[best_move_idx]
            print(f"MCTS Bot selected move with confidence {confidence:.2%}")
        
        return move


def make_mcts_bot(mcts_simulations=100, verbose=False):
    """Factory function to create MCTS bot.
    
    Args:
        mcts_simulations: Number of MCTS simulations per move (when no time limit)
        verbose: Print debug information
        
    Returns:
        MCTSBot instance
    """
    return MCTSBot(mcts_base_simulations=mcts_simulations, verbose=verbose)


if __name__ == "__main__":
    # Test the bot
    import sys
    sys.path.insert(0, '/home/redmond/Projects/Onitama')
    from Boardstate import Boardstate
    
    print("Testing MCTS Bot...")
    bot = MCTSBot(mcts_base_simulations=10, verbose=True)
    
    board = Boardstate()
    print(f"\nInitial board state:")
    print(f"  {board.turn_colour()}'s turn")
    
    print(f"\n1. Test with fixed simulations:")
    move = bot.request_move(board)
    print(f"   Selected move: {move}")
    
    print(f"\n2. Test with 1 second time limit:")
    board2 = Boardstate()
    move2 = bot.request_move(board2, time_limit=1.0)
    print(f"   Selected move: {move2}")
