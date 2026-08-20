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
        red_start = None,
        positions = None
        ):

        if any([red_cards is None, blue_cards is None, trans_card is None]):
            card_sample = random.sample(Deck.DECK, 5)
            red_cards = (card_sample[0], card_sample[1])
            blue_cards = (card_sample[2], card_sample[3])
            trans_card = card_sample[4]

        if red_start is None:
            red_start = bool(random.randint(0,1))

        if positions is not None:
            # Directly use provided positions list (should be length 10)
            self.positions = positions.copy()
        else:
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
        """Return a custom shallow/deep copy of this boardstate."""
        # Cards and trans_card are assumed to be immutable or not mutated in-place
        return Boardstate(
            red_cards=self.red_cards,
            blue_cards=self.blue_cards,
            trans_card=self.trans_card,
            turn_num=self.turn_num,
            red_start=self.red_start,
            positions=self.positions.copy()
        )

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

    # pretty print board
    def create_matrix(self):
        # Initialise matrix for containing piece locations
        matrix = [[" "," "," "," "," "] for i in range(5)]

        # Loop through pieces on the board, replace corresponsing matrix entry with appropriate character
        for num, piece in enumerate(self.positions):

            if piece is None: continue # Ignore piece if captured

            if num == 0: matrix[piece[1]][piece[0]] = "R"
            elif num <= 4: matrix[piece[1]][piece[0]] = "r"
            elif num == 5: matrix[piece[1]][piece[0]] = "B"
            else: matrix[piece[1]][piece[0]] = "b"
        
        # Initialise first line of full_matrix
        full_matrix = [
            "+---+---+---+---+---+"
        ]

        bar = " | " # Prepare string to make f string list join work

        # successively fill out the rest of full_matrix
        for i in range(5):
            full_matrix.append(f"| {bar.join(matrix[i])} |")
            full_matrix.append("+---+---+---+---+---+")
        
        return full_matrix

    def board_str(self, side=None):
        # Prepare the move matrix for each card in play, to be displayed
        # red_cards has RED's cards (positions 0-4)
        # blue_cards has BLUE's cards (positions 5-9)
        matrix_red_0 = self.red_cards[0].matrix
        matrix_red_1 = self.red_cards[1].matrix
        matrix_blue_0 = self.blue_cards[0].matrix
        matrix_blue_1 = self.blue_cards[1].matrix
        matrix_trans = self.trans_card.matrix

        # Prepare string
        s = ''

        if side is None: 
            side = self.turn_colour()

        if side == 'RED': # Red's perspective

            for i in range(5): # Prints Blue's cards
                s += f" {matrix_blue_1[i]}   {matrix_blue_0[i]} \n" if i == 0 else f" {matrix_blue_1[i][::-1]}   {matrix_blue_0[i][::-1]} \n"

            # Prints play area
            create_matrix_result = self.create_matrix()
            for num, row in enumerate(reversed(create_matrix_result)): # Prints row numbers
                s += " " if num%2 == 0 else str(4-(num//2)) 
                
                # Prints rows from self matrix, followed by trans card for central rows
                if 3 <= num and num <= 7:
                    s += f"{row[::-1]} {matrix_trans[7-num]}\n"
                else: s += f"{row[::-1]}\n"
            s += "   a   b   c   d   e  \n"

            for i in range(5): # Prints Red's cards
                s += f" {matrix_red_0[4-i]}   {matrix_red_1[4-i]} \n"

        else:   # Blue's perspective

            for i in range(5): # Prints Red's cards
                s += f" {matrix_red_1[i]}   {matrix_red_0[i]} \n" if i == 0 else f" {matrix_red_1[i][::-1]}   {matrix_red_0[i][::-1]} \n"

            # Prints play area
            create_matrix_result = self.create_matrix()
            for num, row in enumerate(create_matrix_result): # Prints row numbers
                s += " " if num%2 == 0 else str(num//2)

                # Prints rows from self matrix, followed by trans card for central rows
                if 3 <= num and num <= 7: 
                    s += f"{row} {matrix_trans[7-num]}\n"
                else: s += f"{row}\n"
            s += "   e   d   c   b   a  \n"

            for i in range(5): # Prints Blue's cards
                s += f" {matrix_blue_0[4-i]}   {matrix_blue_1[4-i]} \n"
        return s

    def validate_move(self, move_coords):
        # move_coords: ((x0,y0),(x1,y1))
        # Determine whose turn in absolute terms
        turn_is_red = (self.turn_colour() == "RED")

        if turn_is_red:
            pieces = self.positions[0:5]
            cards = self.red_cards
            mul = 1
        else:
            pieces = self.positions[5:10]
            cards = self.blue_cards
            mul = -1

        if move_coords[0] not in pieces:
            print("Coordinates do not specify a friendly piece.")
            return None

        card_move = (-mul*move_coords[1][0]+mul*move_coords[0][0], mul*move_coords[1][1]-mul*move_coords[0][1])

        # card.moveset in Deck.Card
        if card_move not in cards[0].moveset and card_move not in cards[1].moveset:
            print("Move not in hand.")
            return None

        if move_coords[1] in pieces:
            print("Move cannot capture a friendly piece.")
            return None

        if card_move in cards[0].moveset and card_move in cards[1].moveset:
            # ambiguous; ask user which card to use
            while True:
                response = input("Move is possible with either card. Name the desired card: ")
                if response == cards[0].name or response == cards[1].name:
                    break
                print("That is not the name of one of your cards.")
            
            card_used = 0 if response == cards[0].name else 1
        else:
            card_used = 0 if card_move in cards[0].moveset else 1

        # Determine global card idx: use Card.idx attribute
        return (move_coords[0], move_coords[1], cards[card_used].idx)

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

    def possible_moves_search_optimised(self, separate=False):
        """Return legal moves with captures prioritized.
        
        If separate=False (default): returns list [captures] + [other_moves]
        If separate=True: returns tuple (captures, other_moves)
        """
        captures = []
        other_moves = []
        if self.turn_colour() == "RED":
            current_cards = self.red_cards
            red_positions = self.positions[0:5]
            blue_positions = self.positions[5:10]
            master = True
            for position in red_positions:
                if position is None: continue
                for card in current_cards:
                    for card_move in card.moveset:
                        destination = (position[0] - card_move[0], card_move[1] + position[1])
                        if destination[0] < 0 or destination[0] > 4: continue
                        if destination[1] < 0 or destination[1] > 4: continue
                        if destination in red_positions: continue
                        if destination == self.positions[5]: 
                            winning_move = [(position,destination,card.idx)]
                            return (winning_move, []) if separate else winning_move
                        if master and destination == (2,4): 
                            winning_move = [(position,destination,card.idx)]
                            return (winning_move, []) if separate else winning_move
                        elif destination in blue_positions: captures.append((position,destination,card.idx))
                        else: other_moves.append((position,destination,card.idx))
                master = False
        else:
            current_cards = self.blue_cards
            blue_positions = self.positions[5:10]
            red_positions = self.positions[0:5]
            master = True
            for position in blue_positions:
                if position is None: continue
                for card in current_cards:
                    for card_move in card.moveset:
                        destination = (position[0] + card_move[0], -card_move[1] + position[1])
                        if destination[0] < 0 or destination[0] > 4: continue
                        if destination[1] < 0 or destination[1] > 4: continue
                        if destination in blue_positions: continue
                        if destination == self.positions[0]: 
                            winning_move = [(position,destination,card.idx)]
                            return (winning_move, []) if separate else winning_move
                        if master and destination == (2,0): 
                            winning_move = [(position,destination,card.idx)]
                            return (winning_move, []) if separate else winning_move
                        elif destination in red_positions: captures.append((position,destination,card.idx))
                        else: other_moves.append((position,destination,card.idx))
                master = False

        if separate:
            return captures, other_moves
        return captures + other_moves

    def game_stage(self):
        num_blue_students = sum(1 for p in self.positions[6:10] if p is not None)
        num_red_students = sum(1 for p in self.positions[1:5] if p is not None)
        min_students = min(num_blue_students, num_red_students)
        if min_students >= 4: return "OPENING"
        elif min_students >= 2: return "MIDGAME"
        else: return "ENDGAME"
