import random
import copy
from onitama.Deck import Deck
from onitama.Card import *


class Boardstate:

    def __init__(self, 
        red_master   = (2, 0),
        red_students = [(0, 0), (1, 0), (3, 0), (4, 0)],
        blue_master     = (2, 4),
        blue_students   = [(0, 4), (1, 4), (3, 4), (4, 4)],
        red_cards = None,
        blue_cards = None,
        trans_card = None,
        turn_num = 0,
        red_start = None
        ):

        if any([red_cards is None, blue_cards is None, trans_card is None]):
            card_sample = random.sample(Deck.DECK, 5)
            red_cards = (card_sample[0], card_sample[1])
            blue_cards = (card_sample[2], card_sample[3])
            trans_card = card_sample[4]

        if red_start is None:
            red_start = bool(random.randint(0,1))

        # Store positions as 10-element list: [r_master, r_s1..s4, b_master, b_s1..s4]
        self.positions = [red_master]
        self.positions.extend(red_students)
        self.positions.append(blue_master)
        self.positions.extend(blue_students)
        
        self.red_cards = red_cards
        self.blue_cards = blue_cards
        self.trans_card = trans_card
        self.turn_num = turn_num
        self.red_start = red_start

    # Helpers to convert between flat arrays and coordinate tuples used in legacy code
    @staticmethod
    def idx_to_coord(idx):
        return (idx % 5, idx // 5)

    @staticmethod
    def coord_to_idx(coord):
        return coord[0] + 5*coord[1]

    def turn_colour(self, opnt_colour=False):
        return "RED" if self.red_start ^ (self.turn_num % 2 != opnt_colour) else "BLUE"

    def copy(self):
        """Return a deep copy of this boardstate."""
        return copy.deepcopy(self)

    def is_won(self):
        if self.positions[0] == (2,4):
            return ("RED", "STREAM")
        if self.positions[5] == (2,0):
            return ("BLUE", "STREAM")
        if self.positions[0] is None:
            return ("BLUE", "STONE")
        if self.positions[5] is None:
            return ("RED", "STONE")
        return None

    def execute_move(self, move):
        # move: (start_coord, dest_coord, card_idx)
        # For backwards compatibility execute_move delegates to apply_move
        self.apply_move(move)

    def apply_move(self, move):
        """Apply a move in-place and return an undo record.

        Move format: (start_coord, dest_coord, card_idx)
        Undo record contains enough state to restore the board.
        """
        # record previous state
        prev_turn = self.turn_num
        prev_red_cards = self.red_cards
        prev_blue_cards = self.blue_cards
        prev_trans = self.trans_card

        side = self.turn_colour()

        # swap used card with trans_card for current player
        if side == "RED":
            cards = list(self.red_cards)
            if cards[0].idx == move[2]:
                used_card = cards[0]
                cards[0] = self.trans_card
            else:
                used_card = cards[1]
                cards[1] = self.trans_card
            self.trans_card = used_card
            self.red_cards = tuple(cards)
        else:
            cards = list(self.blue_cards)
            if cards[0].idx == move[2]:
                used_card = cards[0]
                cards[0] = self.trans_card
            else:
                used_card = cards[1]
                cards[1] = self.trans_card
            self.trans_card = used_card
            self.blue_cards = tuple(cards)

        # perform capture (record captured piece index if any)
        try:
            captured_index = self.positions.index(move[1])
            self.positions[captured_index] = None
        except ValueError:
            captured_index = None

        # move piece (record start index)
        try:
            start_index = self.positions.index(move[0])
            self.positions[start_index] = move[1]
        except ValueError:
            start_index = None

        # advance turn
        self.turn_num += 1

        return {
            'prev_turn': prev_turn,
            'prev_red_cards': prev_red_cards,
            'prev_blue_cards': prev_blue_cards,
            'prev_trans': prev_trans,
            'captured_index': captured_index,
            'start_index': start_index,
            'move': move
        }

    def undo_move(self, undo_record):
        """Undo a previously applied move using the undo record returned by apply_move."""
        move = undo_record['move']

        # restore moved piece
        if undo_record['start_index'] is not None:
            self.positions[undo_record['start_index']] = move[0]

        # restore captured piece if any
        if undo_record['captured_index'] is not None:
            self.positions[undo_record['captured_index']] = move[1]

        # restore cards and trans
        self.red_cards = undo_record['prev_red_cards']
        self.blue_cards = undo_record['prev_blue_cards']
        self.trans_card = undo_record['prev_trans']

        # restore turn
        self.turn_num = undo_record['prev_turn']

    # Returns a list of legal moves given a position.
    def possible_moves(self):
        if self.turn_colour() == "RED":
            current_cards = self.red_cards
            current_pieces = self.positions[0:5]
            mul = 1
        else:
            current_cards = self.blue_cards
            current_pieces = self.positions[5:10]
            mul = -1

        possible_moves = []
        for position in current_pieces:
            if position is None: continue

            for card in current_cards:
                for card_move in card.moveset:
                    destination = (position[0] - mul*card_move[0], mul*card_move[1] + position[1])
                    if destination[0] < 0 or destination[0] > 4: continue
                    if destination[1] < 0 or destination[1] > 4: continue
                    if destination in current_pieces: continue
                    possible_moves.append((position,destination,card.idx))

        return possible_moves
