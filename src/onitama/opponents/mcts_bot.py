import random
import time
import math
from onitama.opponents import ab_search_bot

OPT_TEMP = 0.7
OPT_EXPLORE = 700.0

class MCTSNode:
    """A node in the Monte Carlo Tree Search tree."""
    
    def __init__(self, board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0.0  # cumulative reward
        self.untried_moves = board.possible_moves_search_optimised()
        
    def ucb(self, exploration=10.0):
        """Calculate the UCB (Upper Confidence Bound) for this node."""

        # Potentially causes undesirable amount of branching?
        # Ensures each initial move is explored at least once.
        if self.visits == 0:
            return float('inf')
        
        # Get the parent's turn color to determine what's best for the player making this move,
        # rather than the player who will play after this move.
        mul = 1 if self.parent.board.turn_colour() == "RED" else -1
        
        exploitation = mul * self.value / self.visits
        exploration_term = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration_term
    
    def select_child(self, exploration=10.0):
        """Select the child with the highest UCB value."""
        return max(self.children, key=lambda child: child.ucb(exploration))
    
    def expand(self):
        """Expand one untried move from this node."""
        move = self.untried_moves.pop(0)
        child_board = self.board.copy()
        child_board.apply_move(move)
        child = MCTSNode(child_board, parent=self, move=move)
        self.children.append(child)
        return child
    
    def is_fully_expanded(self):
        """Check if all moves have been tried."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self):
        """Check if this is a terminal node."""
        return self.board.is_won() is not None


class MCTSBot:
    def __init__(self, time_limit=1.0, temperature=OPT_TEMP, exploration=OPT_EXPLORE):
        """Initialize MCTS bot.
        
        Args:
            time_limit: Time limit in seconds for search (default 1.0)
            temperature: Temperature parameter for softmax in rollouts
                        - Lower values favor higher-scoring moves (harder max)
                        - Higher values approach uniform randomness
        """
        self.time_limit = time_limit
        self.temperature = temperature
        self.evaluator = ab_search_bot.ABSearchBot()
        self.exploration = exploration
        
        # Instrumentation counters
        self.nodes_created = 0
        self.rollouts_evaluated = 0
        self.board_copies = 0
        # cumulative time spent performing rollouts (seconds)
        self.rollout_time = 0.0
    
    def request_move(self, board, time_limit=None):
        """Request a move using MCTS within the time limit.
        
        Args:
            board: Current boardstate
            time_limit: Time limit in seconds (uses time_limit if None)
        
        Returns:
            A move tuple (start_coord, dest_coord, card_idx)
        """
        if time_limit is None:
            time_limit = self.time_limit
        
        # Reset counters
        self.nodes_created = 1
        self.rollouts_evaluated = 0
        self.board_copies = 0
        self.rollout_time = 0.0
        
        deadline = time.time() + time_limit
        root = MCTSNode(board)
        
        # Run MCTS iterations until deadline
        while time.time() < deadline:
            self._mcts_iteration(root, deadline)
        
        # Return the move with the most visits
        if not root.children:
            # No moves explored, return random move
            moves = board.possible_moves_search_optimised()
            return random.choice(moves) if moves else None
        
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.move
    
    def _mcts_iteration(self, root, deadline):
        """Perform one iteration of MCTS: selection, expansion, simulation, backpropagation."""
        # Selection: traverse tree using UCB
        node = root
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.select_child(exploration=self.exploration)
        
        # Expansion: if node is not terminal and not fully expanded, expand one child
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()
            self.nodes_created += 1
        
        # Simulation: run a playout from current node using softmax-weighted moves
        # measure rollout duration separately
        start_rollout = time.time()
        reward = self._rollout(node.board, deadline)
        self.rollout_time += time.time() - start_rollout
        
        # Backpropagation: update values and visit counts up the tree
        self._backpropagate(node, reward)
    
    def _rollout(self, board, deadline):
        """Run a rollout from the current board using softmax-weighted move selection.
        
        Returns the evaluation of the final position (RED-positive).
        """
        sim_board = board.copy()
        self.board_copies += 1
        
        while sim_board.is_won() is None:
            # Check deadline
            if time.time() >= deadline:
                break

            mul = 1 if sim_board.turn_colour() == "RED" else -1

            moves = sim_board.possible_moves_search_optimised()
            
            if self.temperature is None:
                # If temperature is None, select uniformly at random without evaluating positions
                selected_move = random.choice(moves)
            else:
                # Evaluate all possible moves
                move_scores = []
                for move in moves:
                    undo_record = sim_board.apply_move(move)
                    try:
                        score = mul * self.evaluator._evaluate_position(sim_board)
                        move_scores.append(score)
                    finally:
                        sim_board.undo_move(undo_record)
                
                # Select next move using softmax with temperature
                selected_move = self._select_move_softmax(moves, move_scores)
                
            # Apply selected move
            sim_board.apply_move(selected_move)
            
            self.rollouts_evaluated += 1
        
        # Evaluate final position
        final_eval = self.evaluator._evaluate_position(sim_board)
        return final_eval
    
    def _select_move_softmax(self, moves, scores):
        """Select a move using softmax probabilities based on evaluation scores.
        
        Args:
            moves: List of possible moves
            scores: List of evaluation scores for each move (RED-positive)
        
        Returns:
            Selected move based on softmax probabilities
        """
        if not moves:
            return None
        
        # Handle edge case of single move
        if len(moves) == 1:
            return moves[0]
        
        # Find max score for numerical stability
        max_score = max(scores)
        
        # Compute softmax probabilities with temperature
        exp_scores = []
        for score in scores:
            # Temperature scaling: divide by temperature before exp
            exp_val = math.exp((score - max_score) / self.temperature)
            exp_scores.append(exp_val)
        
        # Normalize to get probabilities
        sum_exp = sum(exp_scores)
        probabilities = [p / sum_exp for p in exp_scores]
        
        # Select move based on probabilities
        r = random.random()
        cumulative = 0.0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return moves[i]
    
    
    def _backpropagate(self, node, reward):
        """Backpropagate the reward up the tree."""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def get_stats(self):
        """Return instrumentation statistics from the last move request.
        
        Returns: dict with keys:
            - nodes_created: Number of MCTS nodes created
            - rollouts_evaluated: Number of rollouts evaluated during rollouts
            - board_copies: Number of times board.copy() was called
        """
        return {
            'nodes_created': self.nodes_created,
            'rollouts_evaluated': self.rollouts_evaluated,
            'board_copies': self.board_copies,
            'time_in_rollouts': self.rollout_time
        }

