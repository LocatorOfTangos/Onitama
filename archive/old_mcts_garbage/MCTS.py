"""
MCTS.py

Skeleton implementation of MCTS training components for Onitama.
- Uses LeanBoardstate as the primary state representation for training.
- Provides a NeuralNet placeholder with `predict(lean_state)` -> (policy, value).
- Provides MCTS search loop (selection, expansion, backup) using the net.
- Provides ReplayBuffer and SelfPlayTrainer scaffolding that runs self-play
  with MCTS and collects training examples.

This module is intentionally self-contained and does not modify `main.py` or
`Boardstate.py`. The training code expects `LeanBoardstate` inputs.
"""

import math
import random
from collections import deque, defaultdict
import numpy as np

try:
    import LeanBoardstate
except Exception:
    # Keep import lazy-friendly for tests; consumer will provide LeanBoardstate
    LeanBoardstate = None


class NeuralNet:
    """Lightweight NumPy MLP used as a wiring placeholder for training.

    - `predict(lean_state, legal_moves=None)` returns (priors_dict, value_scalar).
      If `legal_moves` is provided, priors are uniform initially.
    - `train(examples)` performs a tiny SGD step on the value head (placeholder).

    This class is intended to be replaced by a proper PyTorch/TensorFlow model
    for production training; it's only to wire up the rest of the pipeline.
    """
    def __init__(self, input_size=5*5*4 + 16*3 + 1, hidden_sizes=(128,)*8, output_size=67, seed=42):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        rnd = random.Random(seed)
        
        # Initialize weights and biases for hidden layers
        sizes = (input_size,) + hidden_sizes
        self.W = [np.array([[rnd.uniform(-0.1, 0.1) for _ in range(sizes[i])] for __ in range(sizes[i+1])], dtype=float) for i in range(len(sizes)-1)]
        self.b = [np.array([0.0]*sizes[i+1], dtype=float) for i in range(len(sizes)-1)]
        
        # Output layer: maps from last hidden layer to output_size
        last_hidden = hidden_sizes[-1]
        self.W_out = np.array([[rnd.uniform(-0.1, 0.1) for _ in range(last_hidden)] for __ in range(output_size)], dtype=float)
        self.b_out = np.array([0.0]*output_size, dtype=float)

    def _state_to_vector(self, state):
        """Convert a LeanBoardstate object to a flat numpy vector.

        The state has pieces as coordinate tuples (lists of positions) and cards as Card objects.
        We convert positions to 5x5 arrays where:
        - 'act' (actor) is the current player (determined by turn_num and red_start)
        - 'opp' (opponent) is the other player
        Card vectors are binary length-16 arrays indicating which cards the player has.
        """
        if isinstance(state, np.ndarray):
            return state.flatten().astype(float)

        # Determine whose turn it is
        turn_num = state.turn_num
        red_start = state.red_start
        is_red_turn = red_start ^ (turn_num % 2 != 0)

        # Convert positions list to arrays
        # positions is [red_master, red_s1..s4, blue_master, blue_s1..s4]
        def coord_to_board(coord):
            """Convert a coordinate tuple to a 5x5 board with 1 at that position."""
            board = np.zeros((5, 5), dtype=float)
            if coord is not None:
                x, y = coord
                board[y, x] = 1.0
            return board

        def coords_list_to_board(coords_list):
            """Convert a list of coordinate tuples to a 5x5 board with 1s at those positions."""
            board = np.zeros((5, 5), dtype=float)
            for coord in coords_list:
                if coord is not None:
                    x, y = coord
                    board[y, x] = 1.0
            return board

        positions = state.positions
        red_master = coord_to_board(positions[0])
        red_students = coords_list_to_board(positions[1:5])
        blue_master = coord_to_board(positions[5])
        blue_students = coords_list_to_board(positions[6:10])

        # Determine actor and opponent based on whose turn it is
        if is_red_turn:
            act_board = np.concatenate([red_master.flatten(), red_students.flatten()])
            opp_board = np.concatenate([blue_master.flatten(), blue_students.flatten()])
            act_cards = state.red_cards
            opp_cards = state.blue_cards
        else:
            act_board = np.concatenate([blue_master.flatten(), blue_students.flatten()])
            opp_board = np.concatenate([red_master.flatten(), red_students.flatten()])
            act_cards = state.blue_cards
            opp_cards = state.red_cards

        # Cards: create binary vectors of length 16
        def card_vector(cards):
            vec = np.zeros(16, dtype=float)
            for card in cards:
                vec[int(card.idx)] = 1.0
            return vec

        act_cards_vec = card_vector(act_cards)
        opp_cards_vec = card_vector(opp_cards)

        # Trans card: binary vector of length 16
        trans_vec = np.zeros(16, dtype=float)
        trans_vec[int(state.trans_card.idx)] = 1.0

        # Turn: actual turn number
        turn = np.array([float(state.turn_num)], dtype=float)

        vec = np.concatenate([
            act_board.flatten(),
            opp_board.flatten(),
            act_cards_vec.flatten(),
            opp_cards_vec.flatten(),
            trans_vec.flatten(),
            turn.flatten()
        ])
        
        # Ensure vector has expected input size, pad or trim
        if vec.size < self.input_size:
            vec = np.pad(vec, (0, self.input_size - vec.size))
        elif vec.size > self.input_size:
            vec = vec[:self.input_size]
        return vec

    def _move_to_indices(self, move):
        """Convert a move (start_coord, end_coord, card_idx) to output indices.
        
        Returns (start_idx, end_idx, card_idx) where:
        - start_idx: position in [0:25] for starting position
        - end_idx: position in [25:50] for ending position
        - card_idx: position in [50:66] for card
        """
        start_coord, end_coord, card_idx = move
        start_x, start_y = start_coord
        end_x, end_y = end_coord
        
        start_pos_idx = start_y * 5 + start_x
        end_pos_idx = 25 + (end_y * 5 + end_x)
        card_output_idx = 50 + int(card_idx)
        
        return start_pos_idx, end_pos_idx, card_output_idx

    def _get_legal_moves(self, state):
        """Get all legal moves for a state and convert to output indices."""
        moves = state.possible_moves()
        return [self._move_to_indices(move) for move in moves]

    def _policy_from_output(self, output, legal_move_indices):
        """Convert raw output to a probability distribution over legal moves.
        
        Aggregates the three sub-outputs for each move and applies softmax over legal moves.
        Returns a dictionary mapping move indices (tuples) to probabilities.
        """
        if not legal_move_indices:
            return {}
        
        # For each legal move, compute score from start + end + card outputs
        move_scores = []
        for start_idx, end_idx, card_idx in legal_move_indices:
            # Score is the product (or could be sum) of the three outputs
            score = output[start_idx] * output[end_idx] * output[card_idx]
            move_scores.append(score)
        
        move_scores = np.array(move_scores)
        # Apply softmax to get probabilities
        move_scores = np.maximum(move_scores, 0)  # Ensure non-negative
        move_scores = move_scores + 1e-8  # Avoid division by zero
        probs = move_scores / np.sum(move_scores)
        
        policy_dict = {tuple(move_idx): float(prob) 
                      for move_idx, prob in zip(legal_move_indices, probs)}
        return policy_dict
    
    def _forward(self, x):
        a = x
        # Pass through hidden layers with ReLU
        for W, b in zip(self.W, self.b):
            a = np.dot(W, a) + b
            a = np.maximum(a, 0)  # ReLU
        
        # Output layer
        output = np.dot(self.W_out, a) + self.b_out
        
        # Apply activations to output:
        # - First 66 nodes (positions + cards): ReLU (positive)
        # - Last 1 node (position value): sigmoid scaled to [-1, 1]
        output[:66] = np.maximum(output[:66], 0)  # ReLU for positions and card
        output[66] = np.tanh(output[66])  # tanh for position value in [-1, 1]
        
        return a, output

    def predict(self, state):
        """Return (policy_dict, value).
        
        policy_dict: maps move tuples (start_idx, end_idx, card_idx) to probabilities.
        value: position value in [-1, 1].
        """
        x = self._state_to_vector(state)
        _, output = self._forward(x)
        
        legal_move_indices = self._get_legal_moves(state)
        policy = self._policy_from_output(output, legal_move_indices)
        
        value = float(output[66])
        return policy, value

    def train(self, examples, lr=1e-3, debug=False):
        """Train the network on examples.

        examples: list of (state, pi, z) where:
          - state: LeanBoardstate
          - pi: dict mapping move tuples (start_idx, end_idx, card_idx) to probabilities (the empirical policy from MCTS)
          - z: outcome value in [-1, 1]
        
        Optimizes policy loss (KL divergence) and value loss (MSE on outcome).
        """
        if not examples:
            return
        
        grads_W_out = np.zeros_like(self.W_out)
        grads_b_out = np.zeros_like(self.b_out)
        count = 0
        total_loss = 0.0
        
        for state, pi, z in examples:
            x = self._state_to_vector(state)
            a, output = self._forward(x)
            
            # Get legal moves (as indices)
            legal_move_indices = self._get_legal_moves(state)
            if not legal_move_indices:
                continue
            
            # Compute predicted policy from network output
            pred_policy = self._policy_from_output(output, legal_move_indices)
            
            # Policy loss: KL divergence between true and predicted policy
            # KL(pi || pred_policy) = sum_moves pi[move] * (log(pi[move]) - log(pred_policy[move]))
            for move_idx in legal_move_indices:
                move_tuple = tuple(move_idx)
                true_prob = pi.get(move_tuple, 1e-8)  # True policy probability
                pred_prob = max(pred_policy.get(move_tuple, 1e-8), 1e-10)  # Clamp to avoid division issues
                
                # Contribution to loss
                kl_loss = true_prob * (np.log(true_prob + 1e-10) - np.log(pred_prob))
                total_loss += kl_loss
                
                # KL gradient: d/d(pred_prob) = -true_prob / pred_prob
                kl_grad = -true_prob / (pred_prob + 1e-10)
                
                start_idx, end_idx, card_idx = move_idx
                
                # Chain rule through product structure: output[start] * output[end] * output[card]
                # d/d(output[start]) of product = output[end] * output[card]
                # d product / d(output[start]) contributes to gradient
                grads_W_out[start_idx] += kl_grad * output[end_idx] * output[card_idx] * a
                grads_W_out[end_idx] += kl_grad * output[start_idx] * output[card_idx] * a
                grads_W_out[card_idx] += kl_grad * output[start_idx] * output[end_idx] * a
                grads_b_out[start_idx] += kl_grad * output[end_idx] * output[card_idx]
                grads_b_out[end_idx] += kl_grad * output[start_idx] * output[card_idx]
                grads_b_out[card_idx] += kl_grad * output[start_idx] * output[end_idx]
            
            # Value loss: MSE on node 66
            v = output[66]
            value_err = (v - z)
            total_loss += value_err ** 2
            value_grad_scale = 2 * value_err * (1 - v*v)  # Through tanh
            grads_W_out[66] += value_grad_scale * a
            grads_b_out[66] += value_grad_scale
            
            count += 1
        
        if count > 0:
            grads_W_out /= count
            grads_b_out /= count
            
            if debug:
                print(f"  Average loss: {total_loss / count}")
                print(f"  Max grad W_out: {np.max(np.abs(grads_W_out))}")
                print(f"  Max grad b_out: {np.max(np.abs(grads_b_out))}")
            
            # SGD update
            self.W_out -= lr * grads_W_out
            self.b_out -= lr * grads_b_out


class MCTSNode:
    def __init__(self, state, parent=None, prior=0.0, move=None):
        self.state = state  # LeanBoardstate instance
        self.parent = parent
        self.children = {}  # move_tuple (start_idx, end_idx, card_idx) -> MCTSNode
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.move = move  # the move tuple that led to this node

    def value(self):
        """Return average value from perspective of the player who created the child."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, net, explore_const=1.0, n_simulations=800):
        self.net = net
        self.explore_const = explore_const
        self.n_simulations = n_simulations

    def search(self, root_state):
        """Run MCTS simulations and return the policy from the root."""
        root = MCTSNode(root_state)
        for _ in range(self.n_simulations):
            node = root
            search_path = [node]

            # Selection: traverse tree using UCB1   
            if not self._is_terminal(node.state):
                policy, value = self._predict(node.state)
                # Create child nodes if not already expanded
                self._expand(node, policy)
                # Select child node to continue search
                move_tuple, node = self._select_child(node)
                search_path.append(node)
            else:
                # Terminal node; evaluate outcome
                value = self._terminal_value(node.state)

            # Backup: propagate value back up the search path
            self._backup(search_path, value)

        # Build policy from visit counts of root's children
        total_visits = sum(child.visit_count for child in root.children.values())
        pi = {}
        if total_visits > 0:
            for move_tuple, child in root.children.items():
                pi[move_tuple] = child.visit_count / total_visits
        return pi

    def _select_child(self, node):
        """Select best child using UCB1."""
        best_score = -float('inf')
        best_move_tuple = None
        best_child = None
        for move_tuple, child in node.children.items():
            u = self.explore_const * math.sqrt(math.log(node.visit_count) / (1 + child.visit_count))
            score = child.value() + u
            if score > best_score:
                best_score = score
                best_move_tuple = move_tuple
                best_child = child
        return best_move_tuple, best_child

    def _expand(self, node, policy):
        """Create child nodes for each legal move with network-predicted priors.
        
        policy: dict mapping move_tuple (start_idx, end_idx, card_idx) to probabilities
        """
        for move_tuple, p in policy.items():
            if move_tuple not in node.children:
                # Apply move to create child state
                child_state = self._apply_move(node.state, move_tuple)
                node.children[move_tuple] = MCTSNode(state=child_state, parent=node, prior=p, move=move_tuple)

    def _backup(self, search_path, value):
        # value is from perspective of the current player at leaf
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # switch perspective for parent

    def _predict(self, state):
        # Use neural net to predict priors and value
        priors, value = self.net.predict(state)
        return priors, value

    def _is_terminal(self, state):
        """Check if the game has ended at this state."""
        result = state.is_won()
        return result is not None

    def _terminal_value(self, state):
        """Get the game outcome from the current player's perspective.
        
        Returns 1.0 if current player won, -1.0 if current player lost.
        """
        if state is None:
            return 0.0
        result = state.is_won()
        if result is None:
            return 0.0
        winner_color, _ = result
        current_color = state.turn_colour()
        # Return 1.0 if current player's color won, -1.0 if opponent won
        return 1.0 if winner_color == current_color else -1.0

    def _apply_move(self, state, move_tuple):
        """Apply a move to a state and return the resulting state.
        
        move_tuple: (start_idx, end_idx, card_idx)
        
        Converts output indices back to original move format and applies it.
        """
        if state is None:
            return None
        
        start_idx, end_idx, card_idx = move_tuple
        # Convert indices back to coordinates
        start_x, start_y = start_idx % 5, start_idx // 5
        end_x, end_y = (end_idx - 25) % 5, (end_idx - 25) // 5
        
        original_move = ((start_x, start_y), (end_x, end_y), card_idx)
        
        # Make a copy and apply the move
        new_state = state.copy()
        new_state.apply_move(original_move)
        return new_state


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, example):
        self.buffer.append(example)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class SelfPlayTrainer:
    def __init__(self, net, mcts_simulations=100, replay_size=10000):
        self.net = net
        self.mcts = MCTS(net, n_simulations=mcts_simulations)
        self.replay = ReplayBuffer(replay_size)

    def generate_self_play(self, starting_lean_state, temperature=1.0):
        """Run a single self-play game using MCTS to produce training examples.

        Returns a list of (lean_state, pi, z) examples where pi is the MCTS
        visit-count policy and z is the game outcome from the perspective of the
        player to move at that lean_state.
        """
        examples = []
        state = starting_lean_state
        
        # Play until terminal state
        while not self.mcts._is_terminal(state):
            # Run MCTS search to get move policy
            pi = self.mcts.search(state)
            
            if not pi:
                # No legal moves (should be caught by terminal check, but safety)
                break
            
            # Store current state and policy
            examples.append((state.copy(), pi, None))
            
            # Sample move from policy with temperature
            # Higher temperature = more uniform, lower = more greedy
            moves = list(pi.keys())
            probs = np.array([pi[m] for m in moves])
            
            if temperature != 1.0:
                # Adjust probabilities: p' = p^(1/T) / sum(p^(1/T))
                probs = probs ** (1.0 / temperature)
                probs = probs / np.sum(probs)
            
            # Sample move according to the (possibly temperature-adjusted) policy
            move_idx = np.random.choice(len(moves), p=probs)
            move = moves[move_idx]
            
            # Apply move to get next state
            state = self.mcts._apply_move(state, move)
        
        # Game has ended; determine outcome
        result = state.is_won()
        if result is not None:
            winner_color, _ = result
            # Fill in z values based on whose turn it was at each step
            for i, (example_state, pi, _) in enumerate(examples):
                current_color = example_state.turn_colour()
                z = 1.0 if winner_color == current_color else -1.0
                examples[i] = (example_state, pi, z)
        
        return examples

    def train(self, num_iterations=1000, games_per_iteration=10, batch_size=32):
        """High-level training loop.

        For each iteration:
        - generate self-play games and store in replay buffer
        - sample minibatches and train neural net
        - periodically evaluate / save model
        """
        for it in range(num_iterations):
            # Generate self-play games
            for g in range(games_per_iteration):
                # Create a fresh starting state for each game
                if LeanBoardstate is None:
                    print("Warning: LeanBoardstate not available")
                    break
                start = LeanBoardstate.Boardstate()  # Create starting position
                examples = self.generate_self_play(start)
                for ex in examples:
                    self.replay.push(ex)
            
            # Train on accumulated examples
            if len(self.replay) > 0:
                batch = self.replay.sample(min(batch_size, len(self.replay)))
                self.net.train(batch)
            
            # Periodically log progress
            if (it + 1) % 10 == 0:
                print(f"Iteration {it + 1}/{num_iterations}, replay buffer size: {len(self.replay)}")
        
        return


if __name__ == "__main__":
    print("MCTS module loaded. This is a training skeleton; integrate LeanBoardstate and a neural net implementation to use it.")