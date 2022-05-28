import random
import re
from random_bot import *

# Initial postitions for a typical game
INITIAL_POSITIONS = [(2,4),(0,4),(1,4),(3,4),(4,4),(2,0),(0,0),(1,0),(3,0),(4,0)]

# Card class represents a card in Onitama
class Card:
    def __init__(self, name, stamp, card_moves):
        self.name = name
        self.stamp = stamp
        self.card_moves = card_moves
        self.matrix = self.create_matrix()

    def create_matrix(self):

        # Initialise matrix for containing possible moves
        matrix = [[" "," "," "," "," "] for i in range(4)]
        matrix[1][2] = "@"
        
        # Loop through moves, replace corresponsing matrix entry with "#"
        for card_move in self.card_moves:
            matrix[card_move[1]+1][card_move[0]+2] = "#"

        # Initialise full_matrix with first line (spaced card name)
        full_matrix = [
            (9-len(self.name))//2*" "+f"{self.name}"+-(-(9-len(self.name))//2)*" "
        ]

        # Prepare string to make f string list join work
        space = " "

        # successively fill out the rest of full_matrix
        for i in range(4):
            full_matrix.append(f"{space.join(matrix[i])}")
        
        return full_matrix

# The Deck contains all cards in Onitama
DECK = [
    Card('tiger', 'blue', [(0,2), (0,-1)]),
    Card('monkey', 'blue', [(1,1),(-1,1),(-1,-1),(1,-1)]),
    Card('dragon', 'red', [(2,1),(-2,1),(-1,-1),(1,-1)]),
    Card('crab', 'blue', [(2,0),(-2,0),(0,1)]),
    Card('mantis', 'red', [(1,1),(-1,1),(0,-1)]),
    Card('frog', 'red', [(-2,0),(-1,1),(1,-1)]),
    Card('elephant', 'red', [(-1,0),(-1,1),(1,1),(1,0)]),
    Card('rooster', 'red', [(-1,0),(-1,-1),(1,1),(1,0)]),
    Card('boar', 'red', [(-1,0),(0,1),(1,0)]),
    Card('ox', 'blue', [(1,0),(0,-1),(0,1)]),
    Card('crane', 'blue', [(1,-1),(-1,-1),(0,1)]),
    Card('eel', 'blue', [(-1,1),(-1,-1),(1,0)]),
    Card('horse', 'red', [(-1,0),(0,-1),(0,1)]),
    Card('cobra', 'red', [(1,1),(1,-1),(-1,0)]),
    Card('goose', 'blue', [(-1,0),(-1,1),(1,-1),(1,0)]),
    Card('rabbit', 'blue', [(2,0),(1,1),(-1,-1)])
]

# The Board class represents a gamestate
class Board:
    def __init__(self, turn=None, positions=INITIAL_POSITIONS, cards=random.sample(DECK,5)):
        self.positions = positions
        self.cards = cards
        if turn is None:
            self.turn = 0 if cards[4].stamp == "red" else 1  # Set game to start on turn 0 or 1 depending on whether RED or BLUE start respectively
        else: self.turn = turn

    def turn_colour(self):
        return "RED" if self.turn % 2 == 0 else "BLUE"

    def matrix(self):

        # Initialise matrix for containing piece locations
        matrix = [[" "," "," "," "," "] for i in range(5)]

        # Loop through pieces on the board, replace corresponsing matrix entry with appropriate character
        for num, piece in enumerate(self.positions):

            if piece == (-1,-1): continue # Ignore piece if captured

            if num == 0: matrix[piece[1]][piece[0]] = "R"
            elif num <= 4: matrix[piece[1]][piece[0]] = "r"
            elif num <= 5: matrix[piece[1]][piece[0]] = "B"
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

    def validate_move(self, move):

        if self.turn % 2 == 0: 
            pieces = self.positions[0:5] # Red's Master and Students
            cards = self.cards[0:2] # Red's cards
            mul = -1 # Multiplier reverses both axes for given moves
        else: 
            pieces = self.positions[5:10] # Blue's Master and Students
            cards = self.cards[2:4] # Blue's  cards
            mul = 1 # Multiplier does not reverse axes

        # Check input moves a friendly piece
        if move[0] not in pieces: 
            print("Coordinates do not specify a friendly piece.")
            return None

        card_move = (-mul*move[1][0]+mul*move[0][0],mul*move[1][1]-mul*move[0][1]) # Convert move to notation used in cards

        # Check move is described by a card in hand
        if  card_move not in cards[0].card_moves and card_move not in cards[1].card_moves: 
            print("Move not in hand.")
            return None

        # Check input does not capture a friendly piece
        if move[1] in pieces:
            print("Move cannot capture a friendly piece.")
            return None

        # Handle extra response when move is possible with either card
        if card_move in cards[0].card_moves and card_move in cards[1].card_moves: 
            
            while True: # Loop to specify desired card
                response = input("Move is possible with either card. Name the desired card: ")

                # Check that response is a card in the player's hand
                if response == cards[0].name or response == cards[1].name: break

                print("That is not the name of one of your cards.")


            # Record card used
            if response == cards[0].name: card_used = 0
            else: card_used = 1
            move.append(cards[card_used])
            return move

        # Record card used
        if card_move in cards[0].card_moves: card_used = 0
        else: card_used = 1
        move.append(cards[card_used])
        return move

    # Updates the board state by executing a move
    def execute_move(self, move):

        self.turn += 1 # Increments turn count

        # Swaps used card with waiting card
        self.cards[self.cards.index(move[2])] = self.cards[4]
        self.cards[4] = move[2]

        # Captures enemy piece, if it exists
        try: self.positions[self.positions.index(move[1])] = (-1,-1)
        except ValueError: pass

        # Updates piece's position
        self.positions[self.positions.index(move[0])] = move[1]

    # Returns type of victory attained in a given board state, None if the game is not won.
    def is_won(self):
        if self.positions[0] == (2,0): return ("RED", "STREAM")
        elif self.positions[5] == (2,4): return ("BLUE", "STREAM")
        elif self.positions[0] == (-1,-1): return ("BLUE", "STONE")
        elif self.positions[5] == (-1,-1): return ("RED", "STONE")
        else: return None

    # Returns subjective game stage evaluation
    def game_stage(self):
        num_red_students = len([1 for student in self.positions[1:5] if student != (-1,-1)])
        num_blue_students = len([1 for student in self.positions[6:10] if student != (-1,-1)])
        min_students = min(num_red_students,num_blue_students)
        if min_students >= 4: return "OPENING"
        elif min_students >= 2: return "MIDGAME"
        else: return "ENDGAME"

    # returns a string representation of the board that is pretty
    def __str__(self):

        # Prepare the move matrix for each card in play, to be displayed
        matrix0 = self.cards[0].matrix
        matrix1 = self.cards[1].matrix
        matrix2 = self.cards[2].matrix
        matrix3 = self.cards[3].matrix
        matrix4 = self.cards[4].matrix

        # Prepare string
        s = ''

        if self.turn % 2 == 0: # Red's turn

            for i in range(5): # Prints Blue's cards
                s += f" {matrix3[i]}   {matrix2[i]} \n" if i == 0 else f" {matrix3[i][::-1]}   {matrix2[i][::-1]} \n"

            # Prints play area
            for num, row in enumerate(self.matrix()): # Prints row numbers
                s += " " if num%2 == 0 else str(num//2)

                # Prints rows from self matrix, followed by 4th card for central rows
                if 3 <= num and num <= 7: 
                    s += f"{row} {matrix4[7-num]}\n"
                else: s += f"{row}\n"
            s += "   a   b   c   d   e  \n"

            for i in range(5): # Prints Red's cards
                s += f" {matrix0[4-i]}   {matrix1[4-i]} \n"

        else:   # Blue's turn

            for i in range(5): # Prints Red's cards
                s += f" {matrix1[i]}   {matrix0[i]} \n" if i == 0 else f" {matrix1[i][::-1]}   {matrix0[i][::-1]} \n"

            # Prints play area
            for num, row in enumerate(reversed(self.matrix())): # Prints row numbers
                s += " " if num%2 == 0 else str(4-(num//2)) 
                
                # Prints rows from self matrix, followed by 4th card for central rows
                if 3 <= num and num <= 7:
                    s += f"{row[::-1]} {matrix4[7-num]}\n"
                else: s += f"{row[::-1]}\n"
            s += "   e   d   c   b   a  \n"

            for i in range(5): # Prints Blue's cards
                s += f" {matrix2[4-i]}   {matrix3[4-i]} \n"
        return s

    # Returns a list of legal moves given a position.
    def possible_moves(self):
        if self.turn_colour() == "RED":
            current_cards = self.cards[0:2] 
            current_pieces = self.positions[0:5]
            mul = -1
        else:
            current_cards = self.cards[2:4] 
            current_pieces = self.positions[5:10]
            mul = 1

        possible_moves = []
        for piece_num, position in enumerate(current_pieces):
            if position == (-1,-1): continue

            for card in current_cards:

                for card_move in card.card_moves:

                    destination = (position[0] - mul*card_move[0], mul*card_move[1] + position[1])

                    if destination[0] not in range(5) or destination[1] not in range(5) or destination in current_pieces: continue
                    possible_moves.append([position,destination,card])

        return possible_moves

    # A simple evaluation function. A positive score favours RED, a negative score favours BLUE
    def evaluate_position(self):
        # Return large evaluation for won positions
        try: return 300 if self.is_won()[0] == "RED" else -300
        except: pass

        score = 0

        # Prioritise the center and add piece values
        for piece in self.positions[1:5]:
            if piece != (-1,-1): score += (CENTER_PRIORITY[piece[0]][piece[1]] + STUDENT_VALUE)
        for piece in self.positions[6:10]:
            if piece != (-1,-1): score -= (CENTER_PRIORITY[piece[0]][piece[1]] + STUDENT_VALUE)

        # Evaluate master positioning
        game_stage = self.game_stage()
        if game_stage == "OPENING":
            score += OPENING_MASTER_POSITIONAL_VALUE[self.positions[0][0]][self.positions[0][1]]
            score -= OPENING_MASTER_POSITIONAL_VALUE[4-self.positions[5][0]][4-self.positions[5][1]]
        elif game_stage == "MIDGAME":
            score += MIDGAME_MASTER_POSITIONAL_VALUE[self.positions[0][0]][self.positions[0][1]]
            score -= MIDGAME_MASTER_POSITIONAL_VALUE[4-self.positions[5][0]][4-self.positions[5][1]]
        else:
            score += ENDGAME_MASTER_POSITIONAL_VALUE[self.positions[0][0]][self.positions[0][1]]
            score -= ENDGAME_MASTER_POSITIONAL_VALUE[4-self.positions[5][0]][4-self.positions[5][1]]    

        return score
            
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

                
def print_victory(victory_type):
    string = f"{victory_type[0]} wins by way of the {victory_type[1]}"
    print("\n"+len(string)*"="+f"\n{string}\n"+len(string)*"="+"\n")
    return

def request_move(board):

    while True: # Loop to request and verify player's move

        move_input = input("Move a piece: ") # Take input on desired move

        # Check regex match
        if not re.match("[a-z]\d[a-z]\d", move_input): 
            print("Could not parse input.")
            continue

        # Parse input to list of integers
        raw_move_coords = []
        for num, char in enumerate(move_input):
            if num%2 == 0: # Parse letters
                raw_move_coords.append(ord(char)-97)
            else:          # Parse digits
                raw_move_coords.append(int(char))

        # Check coordinates are on the board
        if len(raw_move_coords) != 4 or any(x not in range(5) for x in raw_move_coords): 
            print("Invalid Coordinates.")
            continue
        
        # Parses raw_move_coords to coordinate notation used everywhere else
        move_coords = [ (raw_move_coords[0], raw_move_coords[1]), (raw_move_coords[2], raw_move_coords[3]) ] 

        move = board.validate_move(move_coords)

        if move: return move

# Main game loop for two player mode
def two_player_mode():

    print(
'''Two player mode.

Usage: input move by typing start and end coordinates: \'a1b2\'
You may be asked to specify a card. In this case, name the desired card: \'pigeon\'

GAME BEGINS
'''      )

    board = Board() # Set initial boardstate

    while True: # Main game loop

        print(board) # Print boardstate at beginning of any turn

        # Check if the board is won, end game if so
        victory_type = board.is_won()
        if victory_type:
            print_victory(victory_type) # Print victory information
            return

        # Print who's turn it is
        print(f"{board.turn_colour()}\'s turn.")
        print(f"eval = {board.evaluate_position()}")

        # Request a move
        move = request_move(board)

        # Update boardstate by executing move
        board.execute_move(move)

def random_bot_mode():
    player_colour = "RED" if random.randint(0,1) == 0 else "BLUE"

    print(
f'''Random bot mode.

Usage: input move by typing start and end coordinates: \'a1b2\'
You may be asked to specify a card. In this case, name the desired card: \'pigeon\'

You are {player_colour}.

GAME BEGINS
'''      )

    board = Board() # Set initial boardstate

    while True: # Main game loop

        print(board) # Print boardstate at beginning of any turn

        # Check if the board is won, end game if so
        victory_type = board.is_won()
        if victory_type:
            print_victory(victory_type) # Print victory information
            return

        # Print who's turn it is
        print(f"{board.turn_colour()}\'s turn.")

        # Request a move
        if board.turn_colour() == player_colour:
            move = request_move(board)
        else:
            move = request_bot_move(board)
            print('Bot plays: ',chr(move[0][0]+97),move[0][1],chr(move[1][0]+97),move[1][1], sep = '')

        # Update boardstate by executing move
        board.execute_move(move)

if __name__ == "__main__":
    print("\nWelcome to ONITAMA\n")
    response = input("Enter 1 to play locally, 2 to play a bot: ")
    two_player_mode() if response == '1' else random_bot_mode()