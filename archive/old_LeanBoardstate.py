import numpy as np
import random
from Deck import Deck


class LeanBoardstate:
    """
    Lightweight, vectorised Boardstate designed for neural network training and MCTS.
    Uses numpy arrays for efficient computation without positional encoding overhead.
    
    Representation:
    - Board pieces: 5x5 numpy arrays (one for my pieces including both master and students,
      one for opponent pieces including both master and students)
    - Cards: 1-hot encoded arrays of length 16 (16 total cards in deck)
      my_cards: shape (1, 16) with both card indices set to 1
      opnt_cards: shape (1, 16) with both card indices set to 1
      trans_card: shape (16,) for waiting card
    - turn_num: integer track of moves
    """

    def __init__(self,
        my_master=None,
        my_students=None,
        my_cards=None,
        opnt_master=None,
        opnt_students=None,
        opnt_cards=None,
        trans_card=None,
        turn_num=0):
        
        """
        Initialize a LeanBoardstate. If pieces or cards not provided, generates random game.
        
        Args:
            my_master: 5x5 numpy array, 1 at master position, 0 elsewhere
            my_students: 5x5 numpy array, 1s at student positions, 0s elsewhere
            my_cards: tuple of 2 card indices (0-15)
            opnt_master: 5x5 numpy array
            opnt_students: 5x5 numpy array
            opnt_cards: tuple of 2 card indices (0-15)
            trans_card: card index (0-15)
            turn_num: number of moves made
        """
        
        # Initialize board pieces as 5x5 numpy arrays
        if my_master is None:
            my_master = np.zeros((5, 5), dtype=np.uint8)
            my_master[0, 2] = 1  # Blue master starts at (2, 0)
        elif not isinstance(my_master, np.ndarray):
            my_master = np.array(my_master, dtype=np.uint8).reshape(5, 5)
        else:
            my_master = my_master.astype(np.uint8)
            
        if my_students is None:
            my_students = np.zeros((5, 5), dtype=np.uint8)
            my_students[0, [0, 1, 3, 4]] = 1  # Blue students start at (0,0), (1,0), (3,0), (4,0)
        elif not isinstance(my_students, np.ndarray):
            my_students = np.array(my_students, dtype=np.uint8).reshape(5, 5)
        else:
            my_students = my_students.astype(np.uint8)
            
        if opnt_master is None:
            opnt_master = np.zeros((5, 5), dtype=np.uint8)
            opnt_master[4, 2] = 1  # Red master starts at (2, 4)
        elif not isinstance(opnt_master, np.ndarray):
            opnt_master = np.array(opnt_master, dtype=np.uint8).reshape(5, 5)
        else:
            opnt_master = opnt_master.astype(np.uint8)
            
        if opnt_students is None:
            opnt_students = np.zeros((5, 5), dtype=np.uint8)
            opnt_students[4, [0, 1, 3, 4]] = 1  # Red students start at (0,4), (1,4), (3,4), (4,4)
        elif not isinstance(opnt_students, np.ndarray):
            opnt_students = np.array(opnt_students, dtype=np.uint8).reshape(5, 5)
        else:
            opnt_students = opnt_students.astype(np.uint8)
        
        # Initialize cards with 1-hot encoding
        if my_cards is None or opnt_cards is None or trans_card is None:
            card_sample = random.sample(range(16), 5)
            my_cards = tuple(card_sample[0:2])
            opnt_cards = tuple(card_sample[2:4])
            trans_card = card_sample[4]
        
        # Convert card indices to 1-hot arrays (1x16, with both card indices set to 1)
        my_cards_onehot = np.zeros((1, 16), dtype=np.uint8)
        my_cards_onehot[0, my_cards[0]] = 1
        my_cards_onehot[0, my_cards[1]] = 1
        
        opnt_cards_onehot = np.zeros((1, 16), dtype=np.uint8)
        opnt_cards_onehot[0, opnt_cards[0]] = 1
        opnt_cards_onehot[0, opnt_cards[1]] = 1
        
        trans_card_onehot = np.zeros(16, dtype=np.uint8)
        trans_card_onehot[trans_card] = 1
        
        self.my_master = my_master
        self.my_students = my_students
        self.my_cards = my_cards_onehot
        self.opnt_master = opnt_master
        self.opnt_students = opnt_students
        self.opnt_cards = opnt_cards_onehot
        self.trans_card = trans_card_onehot
        self.turn_num = turn_num

    @staticmethod
    def _onehot_to_idx(onehot):
        """Convert 1-hot encoded array to single card index."""
        return np.where(onehot == 1)[0][0]

    @staticmethod
    def _onehot_to_cards(onehot_cards):
        """Convert 1-hot encoded card row (1x16) to tuple of 2 card indices."""
        indices = np.where(onehot_cards[0] == 1)[0]
        return tuple(indices[:2])

    @staticmethod
    def _idx_to_onehot(idx, length=16):
        """Convert card index to 1-hot encoded array."""
        arr = np.zeros(length, dtype=np.uint8)
        arr[idx] = 1
        return arr

    def copy(self):
        """Return a deep copy of this boardstate."""
        return LeanBoardstate(
            my_master=self.my_master.copy(),
            my_students=self.my_students.copy(),
            my_cards=self._onehot_to_cards(self.my_cards),
            opnt_master=self.opnt_master.copy(),
            opnt_students=self.opnt_students.copy(),
            opnt_cards=self._onehot_to_cards(self.opnt_cards),
            trans_card=self._onehot_to_idx(self.trans_card),
            turn_num=self.turn_num
        )

    def is_won_or_lost(self):
        # 1 for win, 0 for loss, None for ongoing
        if self.my_master[4, 2] == 1:
            return 1  # MY wins by STREAM
        # Opponent master at row 0 (my goal) = OPNT wins by STREAM
        if self.opnt_master[0, 2] == 1:
            return 0  # OPNT wins by STREAM
        # My master captured = OPNT wins by STONE
        if np.sum(self.my_master) == 0:
            return 0  # OPNT wins by STONE
        # Opponent master captured = MY wins by STONE
        if np.sum(self.opnt_master) == 0:
            return 1  # MY wins by STONE
        return None

    def _get_piece_positions(self, master, students):
        """Extract (x, y) positions from master and students arrays."""
        positions = []
        # Master
        master_pos = np.where(master == 1)
        if len(master_pos[0]) > 0:
            positions.append((master_pos[1][0], master_pos[0][0]))
        else:
            positions.append(None)
        # Students
        student_pos = np.where(students == 1)
        for i in range(len(student_pos[0])):
            positions.append((student_pos[1][i], student_pos[0][i]))
        # Pad to 5 pieces (1 master + 4 students max)
        while len(positions) < 5:
            positions.append(None)
        return positions[:5]

    def possible_moves(self):
        """
        Generate all legal moves for 'my' pieces (always the current player due to perspective flipping).
        Returns list of moves as ((x0, y0), (x1, y1), card_idx)
        """
        my_positions = self._get_piece_positions(self.my_master, self.my_students)
        my_cards_indices = np.where(self.my_cards[0] == 1)[0]  # All indices with value 1
        opp_positions = self._get_piece_positions(self.opnt_master, self.opnt_students)
        
        # My pieces always move with reversed card coordinates (mul = -1)
        mul = -1

        moves = []
        for pos in my_positions:
            if pos is None:
                continue
            x, y = pos
            for card_idx in my_cards_indices:
                card = Deck.DECK[card_idx]
                for move_delta in card.moveset:
                    dx, dy = move_delta
                    dest_x = x + mul * dx
                    dest_y = y + mul * dy
                    
                    # Check bounds
                    if dest_x < 0 or dest_x > 4 or dest_y < 0 or dest_y > 4:
                        continue
                    # Check friendly fire
                    if (dest_x, dest_y) in my_positions:
                        continue
                    
                    moves.append(((x, y), (dest_x, dest_y), card_idx))
        
        return moves

    def execute_move(self, move):
        """
        Execute a move and flip the board/cards so the opponent becomes 'my'.
        move: ((x0, y0), (x1, y1), card_idx)
        
        This updates state in-place.
        """
        start_pos, end_pos, card_idx = move
        sx, sy = start_pos
        ex, ey = end_pos
        
        # Move piece (it's either in my_master or my_students)
        if self.my_master[sy, sx] == 1:
            self.my_master[sy, sx] = 0
            self.my_master[ey, ex] = 1
        else:
            self.my_students[sy, sx] = 0
            self.my_students[ey, ex] = 1

        # Capture any piece at destination
        if self.opnt_master[ey, ex] == 1:
            self.opnt_master[ey, ex] = 0
        elif self.opnt_students[ey, ex] == 1:
            self.opnt_students[ey, ex] = 0

        # Swap used card with trans_card
        # Store the card being traded out
        old_trans = self.trans_card.copy()
        # The used card is replaced by trans_card
        self.trans_card = np.zeros(16, dtype=np.uint8)
        self.trans_card[card_idx] = 1
        # Set trans_card position in my_cards to 1, used card to 0
        self.my_cards[0, card_idx] = 0
        trans_idx = np.where(old_trans == 1)[0][0]
        self.my_cards[0, trans_idx] = 1

        # Increment turn
        self.turn_num += 1

        # Flip board and cards so opponent becomes 'my'
        self._flip_perspective()

    def _flip_perspective(self):
        """Flip the board and cards so opponent becomes 'my' player."""
        # Flip board: rotate 180 degrees (reverse both axes)
        self.my_master = np.flipud(np.fliplr(self.my_master))
        self.my_students = np.flipud(np.fliplr(self.my_students))
        self.opnt_master = np.flipud(np.fliplr(self.opnt_master))
        self.opnt_students = np.flipud(np.fliplr(self.opnt_students))
        
        # Swap my with opnt
        self.my_master, self.opnt_master = self.opnt_master, self.my_master
        self.my_students, self.opnt_students = self.opnt_students, self.my_students
        self.my_cards, self.opnt_cards = self.opnt_cards, self.my_cards
