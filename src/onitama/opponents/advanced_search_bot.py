import random
import time


class SearchTimeout(Exception):
    pass

#########
MAXDEPTH = 6
#########

# Evaluation constants
CENTER_PRIORITY = [
    [-10,0,10,0,-10],
    [0,10,20,10,0],
    [10,20,30,20,10],
    [0,10,20,10,0],
    [-10,0,10,0,-10]
]

OPENING_MASTER_POSITIONAL_VALUE = [
    [0,10,-20,-60,-100],
    [0,10,-20,-60,-100],
    [0,10,-20,-60,-100],
    [0,10,-20,-60,-100],
    [0,10,-20,-60,-100]
]

MIDGAME_MASTER_POSITIONAL_VALUE = [
    [-20,0,0,-40,-80],
    [-20,0,0,-40,-80],
    [-20,0,0,-40,-80],
    [-20,0,0,-40,-80],
    [-20,0,0,-40,-80]
]

ENDGAME_MASTER_POSITIONAL_VALUE = [
    [-100,-60,20,60,40],
    [-100,-60,20,80,100],
    [-100,-60,20,100,100],
    [-100,-60,20,80,100],
    [-100,-60,20,60,40]
]

STUDENT_VALUE = 50


class AdvancedSearchBot:
    def __init__(self, max_depth=MAXDEPTH):
        self.max_depth = max_depth
        self.expansions = 0
        self.pv_moves = {}  # Principal Variation moves for move ordering
        self.killer_moves = {}  # Killer moves for move ordering
        self.history = {}  # history heuristic: move -> score

    def _game_stage(self, board):
        """Determine game stage from board state."""
        # board.positions layout: [red_master, red_s1..s4, blue_master, blue_s1..s4]
        num_red_students = sum(1 for v in board.positions[1:5] if v is not None)
        num_blue_students = sum(1 for v in board.positions[6:10] if v is not None)
        min_students = min(num_red_students, num_blue_students)
        if min_students >= 4: return "OPENING"
        elif min_students >= 2: return "MIDGAME"
        else: return "ENDGAME"

    def _move_priority(self, move, board, depth, pv_move=None, is_capture=False):
        """Score a move for move ordering. Higher = better."""
        # Check if this is the PV move
        if pv_move and move == pv_move:
            return 10000  # Highest priority
        
        # Captures get high priority (value depends on whether it's a capture)
        if is_capture:
            return 5000
        
        # Check killer moves (non-captures that caused cutoffs)
        killer_key = (depth, board.turn_num)
        if killer_key in self.killer_moves:
            if move in self.killer_moves[killer_key]:
                return 1000
        
        # Base score by destination centrality + history heuristic
        end = move[1]
        base = CENTER_PRIORITY[end[0]][end[1]]
        hist = self.history.get(move, 0)
        return base + hist

    def _score_moves(self, captures, other_moves, board, depth, pv_move=None):
        """Sort moves by priority, with captures first, then other moves.
        
        Captures and other_moves are already separated by possible_moves_search_optimised().
        """
        # Score captures
        scored_caps = [(self._move_priority(m, board, depth, pv_move, is_capture=True), m) for m in captures]
        scored_caps.sort(reverse=True)
        
        # Score other moves
        scored_other = [(self._move_priority(m, board, depth, pv_move, is_capture=False), m) for m in other_moves]
        scored_other.sort(reverse=True)
        
        # Return captures first, then other moves
        return [m for _, m in scored_caps] + [m for _, m in scored_other]

    def _evaluate_position(self, board, depth = 0):
        """Evaluate a board position. Red-positive scoring."""
        try:
            winner = board.is_won()[0]
            depth_discount = depth * 5
            return 1000 - depth_discount if winner == "RED" else -1000 + depth_discount
        except:
            pass

        score = 0
        positions = board.positions

        # red pieces are positions[0:5], blue positions[5:10]
        for piece in positions[1:5]:
            if piece:
                score += (CENTER_PRIORITY[piece[0]][piece[1]] + STUDENT_VALUE)
        for piece in positions[6:10]:
            if piece:
                score -= (CENTER_PRIORITY[piece[0]][piece[1]] + STUDENT_VALUE)

        game_stage = self._game_stage(board)
        if positions[0]:
            rx,ry = positions[0]
            if game_stage == "OPENING":
                score += OPENING_MASTER_POSITIONAL_VALUE[rx][ry]
            elif game_stage == "MIDGAME":
                score += MIDGAME_MASTER_POSITIONAL_VALUE[rx][ry]
            else:
                score += ENDGAME_MASTER_POSITIONAL_VALUE[rx][ry]
        if positions[5]:
            bx,by = positions[5]
            if game_stage == "OPENING":
                score -= OPENING_MASTER_POSITIONAL_VALUE[4-bx][4-by]
            elif game_stage == "MIDGAME":
                score -= MIDGAME_MASTER_POSITIONAL_VALUE[4-bx][4-by]
            else:
                score -= ENDGAME_MASTER_POSITIONAL_VALUE[4-bx][4-by]

        return score

    def request_move(self, board, time_limit=None):
        """Request a move. If time_limit (seconds) is provided, do iterative
        deepening up to self.max_depth and return the best move found within
        that time. If time_limit is None, search to self.max_depth and return
        the result.
        """
        self.expansions = 0
        self.pv_moves = {}
        self.killer_moves = {}

        # No time limit: single search to max_depth
        if time_limit is None:
            return self.recursive_search(board, return_move=True, max_depth=self.max_depth)[1]

        # Iterative deepening with time control
        start_time = time.time()
        deadline = start_time + time_limit
        best_move = None
        depth_limit = 1
        try:
            while True:
                eval_val, move = self.recursive_search(board, return_move=True, max_depth=depth_limit, deadline=deadline)
                if move is not None:
                    best_move = move
                    # Store as PV move for next iteration
                    self.pv_moves[board.turn_num] = move
                # stop if we've already exceeded time after a completed depth
                if time.time() >= deadline:
                    break
                depth_limit += 1
        except SearchTimeout:
            # time expired during a depth; best_move from earlier remains
            pass

        if best_move is None:
            moves = board.possible_moves_search_optimised()
            return random.choice(moves)
        return best_move

    def recursive_search(self, board, depth = 0, alpha = -2000, beta = 2000, return_move = False, max_depth=None, deadline=None):
        self.expansions += 1
        # Check deadline
        if deadline is not None and time.time() >= deadline:
            raise SearchTimeout()
        
        # Get separated move lists
        captures, other_moves = board.possible_moves_search_optimised(separate=True)
        
        # Apply move ordering
        pv_move = self.pv_moves.get(board.turn_num)
        possible_moves = self._score_moves(captures, other_moves, board, depth, pv_move)
        
        best_move_so_far = None

        if board.turn_colour() == "RED":
            eval_to_beat = -2000
            good_move_count = 0
            for i, move in enumerate(possible_moves):
                result = self.simulate(move, board, depth, alpha, beta, max_depth=max_depth, deadline=deadline)

                if alpha > 50 and result >= alpha - 20: # If advantaged and move does not squander advantage
                    good_move_count += 1

                if result > eval_to_beat:
                    eval_to_beat = result
                    best_move_so_far = move
                if eval_to_beat > beta:
                    # Store killer move (non-capture moves cause cutoff)
                    # Killer moves are only non-captures
                    if move not in captures:
                        killer_key = (depth, board.turn_num)
                        if killer_key not in self.killer_moves:
                            self.killer_moves[killer_key] = []
                        if move not in self.killer_moves[killer_key]:
                            self.killer_moves[killer_key].append(move)
                    # Update history heuristic for this move (bigger for shallower cutoffs)
                    boost = (self.max_depth - depth + 1) ** 2
                    self.history[move] = self.history.get(move, 0) + boost
                    break
                alpha = max(alpha, eval_to_beat)

            if not return_move:
                eval_to_beat += max(good_move_count, 3)
                alpha = max(alpha, eval_to_beat)
                
        else:
            eval_to_beat = 2000
            good_move_count = 0
            for i, move in enumerate(possible_moves):
                result = self.simulate(move, board, depth, alpha, beta, max_depth=max_depth, deadline=deadline)

                if beta < -50 and result <= beta + 20:
                    good_move_count += 1

                if result < eval_to_beat:
                    eval_to_beat = result
                    best_move_so_far = move
                if eval_to_beat < alpha:
                    # Store killer move (non-captures only)
                    if move not in captures:
                        killer_key = (depth, board.turn_num)
                        if killer_key not in self.killer_moves:
                            self.killer_moves[killer_key] = []
                        if move not in self.killer_moves[killer_key]:
                            self.killer_moves[killer_key].append(move)
                    # Update history heuristic for this move
                    boost = (self.max_depth - depth + 1) ** 2
                    self.history[move] = self.history.get(move, 0) + boost
                    break
                beta = min(beta, eval_to_beat)
            
            if not return_move:
                eval_to_beat -= max(good_move_count, 3)
                beta = min(beta, eval_to_beat)

        if return_move:
            if best_move_so_far is None:
                best_move_so_far = possible_moves[0] if possible_moves else None
        else:
            best_move_so_far = None

        return eval_to_beat, best_move_so_far
    
    def simulate(self, move, board, depth, alpha, beta, max_depth=None, deadline=None):
        # Use make/undo to avoid deep-copying the board
        if max_depth is None:
            max_depth = self.max_depth

        # check deadline before making move
        if deadline is not None and time.time() >= deadline:
            raise SearchTimeout()

        undo = board.apply_move(move)
        try:
            terminal = board.is_won()
            if depth >= max_depth or terminal:
                return self._evaluate_position(board, depth)
            else:
                return self.recursive_search(board, depth+1, alpha, beta, max_depth=max_depth, deadline=deadline)[0]
        finally:
            board.undo_move(undo)
