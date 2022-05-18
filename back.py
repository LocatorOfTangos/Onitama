from front import *
import random
import re

# Define initial postitions for a typical game
INITIAL_POSITIONS = [(2,4),(0,4),(1,4),(3,4),(4,4),(2,0),(0,0),(1,0),(3,0),(4,0)]

# Card class represents a card in Onitama
class CARD:
    def __init__(self, name, stamp, moves):
        self.name = name
        self.stamp = stamp
        self.moves = moves

    def matrix(self):

        # Initialise matrix for containing possible moves
        matrix = [[" "," "," "," "," "] for i in range(4)]
        matrix[1][2] = "@"
        
        # Loop through moves, replace corresponsing matrix entry with "#"
        for move in self.moves:
            matrix[move[1]+1][move[0]+2] = "#"

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
    CARD('tiger', 'blue', [(0,2), (0,-1)]),
    CARD('monkey', 'blue', [(1,1),(-1,1),(-1,-1),(1,-1)]),
    CARD('dragon', 'red', [(2,1),(-2,1),(-1,-1),(1,-1)]),
    CARD('crab', 'blue', [(2,0),(-2,0),(0,1)]),
    CARD('mantis', 'red', [(1,1),(-1,1),(0,-1)]),
    CARD('frog', 'red', [(-2,0),(-1,1),(1,-1)]),
    CARD('elephant', 'red', [(-1,0),(-1,1),(1,1),(1,0)]),
    CARD('rooster', 'red', [(-1,0),(-1,-1),(1,1),(1,0)]),
    CARD('boar', 'red', [(-1,0),(0,1),(1,0)]),
    CARD('ox', 'blue', [(1,0),(0,-1),(0,1)]),
    CARD('crane', 'blue', [(1,-1),(-1,-1),(0,1)]),
    CARD('eel', 'blue', [(-1,1),(-1,-1),(1,0)]),
    CARD('horse', 'red', [(-1,0),(0,-1),(0,1)]),
    CARD('cobra', 'red', [(1,1),(1,-1),(-1,0)]),
    CARD('goose', 'blue', [(-1,0),(-1,1),(1,-1),(1,0)]),
    CARD('rabbit', 'blue', [(2,0),(1,1),(-1,-1)])
]

# The Board class represents a gamestate
class BOARD:
    def __init__(self, turn, positions, cards):
        self.turn = turn
        self.positions = positions
        self.cards = cards

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

# Returns a board object representing the beginning of an Onitama game. Cards are randomly selected from the Deck
def initialise_board():

    cards = [] # List of cards

    while len(cards)<5: # Choose 5 different random cards from the deck
        num = random.randint(0,len(DECK)-1)
        if DECK[num] not in cards: cards.append(DECK[num])

    turn = 0 if cards[4].stamp == "red" else 1 # Set game to start on turn 0 or 1 depending on whether RED or BLUE start respectively

    return BOARD(turn, INITIAL_POSITIONS, cards)

def print_board(board):

    # Prepare the move matrix for each card in play, to be displayed
    matrix0 = board.cards[0].matrix()
    matrix1 = board.cards[1].matrix()
    matrix2 = board.cards[2].matrix()
    matrix3 = board.cards[3].matrix()
    matrix4 = board.cards[4].matrix()

    if board.turn % 2 == 0: # Red's turn

        for i in range(5): # Prints Blue's cards
            print(f" {matrix3[i]}   {matrix2[i]} ") if i == 0 else print(f" {matrix3[i][::-1]}   {matrix2[i][::-1]} ")

        # Prints play area
        for num, row in enumerate(board.matrix()): # Prints row numbers
            print(" ", end = "") if num%2 == 0 else print(num//2, end = "") 

            # Prints rows from board matrix, followed by 4th card for central rows
            if 3 <= num and num <= 7: 
                print(f"{row} {matrix4[7-num]}")
            else: print(f"{row}")
        print("   a   b   c   d   e  ")

        for i in range(5): # Prints Red's cards
            print(f" {matrix0[4-i]}   {matrix1[4-i]} ")

    else:   # Blue's turn

        for i in range(5): # Prints Red's cards
            print(f" {matrix1[i]}   {matrix0[i]} ") if i == 0 else print(f" {matrix1[i][::-1]}   {matrix0[i][::-1]} ")

        # Prints play area
        for num, row in enumerate(reversed(board.matrix())): # Prints row numbers
            print(" ", end = "") if num%2 == 0 else print(4-(num//2), end = "") 
            
            # Prints rows from board matrix, followed by 4th card for central rows
            if 3 <= num and num <= 7:
                print(f"{row[::-1]} {matrix4[7-num]}")
            else: print(f"{row[::-1]}")
        print("   e   d   c   b   a  ")

        for i in range(5): # Prints Blue's cards
            print(f" {matrix2[4-i]}   {matrix3[4-i]} ")
    print()

def print_victory(victory_type):
    string = f"{victory_type[0]} wins by way of the {victory_type[1]}"
    print("\n"+len(string)*"="+f"\n{string}\n"+len(string)*"="+"\n")
    return

# Updates the board state by executing a move
def execute_move(board, move, card):

    board.turn += 1 # Increments turn count

    # Swaps used card with waiting card
    board.cards[board.cards.index(card)] = board.cards[4]
    board.cards[4] = card

    # Captures enemy piece, if it exists
    try: board.positions[board.positions.index(move[1])] = (-1,-1)
    except ValueError: pass

    # Updates piece's position
    board.positions[board.positions.index(move[0])] = move[1]

# Returns type of victory attained in a given board state, None if the game is not won.
def is_won(board):
    if board.positions[0] == (2,0): return ("RED", "STREAM")
    elif board.positions[5] == (2,4): return ("BLUE", "STREAM")
    elif board.positions[0] == (-1,-1): return ("BLUE", "STONE")
    elif board.positions[5] == (-1,-1): return ("RED", "STONE")
    else: return None

# Main game loop for two player mode
def two_player_mode():

    board = initialise_board() # Set initial boardstate

    print("GAME BEGINS\n")

    while True: # Main game loop

        print_board(board) # Print boardstate at beginning of any turn

        # Check if the board is won, end game if so
        if is_won(board) is not None:
            print_victory(is_won(board)) # Print victory information
            return

        # Print information about who's turn it is, and prepare that player's pieces and cards to be referenced
        if board.turn % 2 == 0: 
            print("RED's turn.")
            pieces = board.positions[0:5] # Red's Master and Students
            cards = board.cards[0:2] # Red's cards
            mul = -1 # Multiplier reverses both axes for given moves
        else: 
            print("BLUE's turn.")
            pieces = board.positions[5:10] # Blue's Master and Students
            cards = board.cards[2:4] # Blue's  cards
            mul = 1 # Multiplier does not reverse axes

        while True: # Loop to request and verify player's move

            move_input = input("Move a piece: ") # Take input on desired move

            # Check regex match
            if not re.match("[a-z]\d[a-z]\d", move_input): 
                print("Could not parse input.")
                continue

            # Parse input to list of integers
            ints = []
            for num, char in enumerate(move_input):
                if num%2 == 0: # Parse letters
                    ints.append(ord(char)-97)
                else:          # Parse digits
                    ints.append(int(char))

            # Check coordinates are on the board
            if len(ints) != 4 or any(x not in range(5) for x in ints): 
                print("Invalid Coordinates.")
                continue
            
            move = [ (ints[0], ints[1]), (ints[2], ints[3]) ] # Parses ints to coordinate notation used everywhere else
            
            # Check input moves a friendly piece
            if move[0] not in pieces: 
                print("Coordinates do not specify a friendly piece.")
                continue

            resultant_move = (-mul*move[1][0]+mul*move[0][0],mul*move[1][1]-mul*move[0][1]) # Convert move to notation used in cards

            # Check move is described by a card in hand
            if  resultant_move not in cards[0].moves and resultant_move not in cards[1].moves: 
                print("Move not in hand.")
                continue

            # Check input does not capture a friendly piece
            if move[1] in pieces:
                print("Move cannot capture a friendly piece.")
                continue

            # Handle extra response when move is possible with either card
            if resultant_move in cards[0].moves and resultant_move in cards[1].moves: 
                
                while True: # Loop to specify desired card
                    response = input("Move is possible with either card. Name the desired card: ")

                    # Check that response is a card in the player's hand
                    if response != cards[0].name and response != cards[1].name:
                        print("That is not the name of one of your cards.")
                        continue
                    break

                # Record card used
                if response == cards[0].name: card_used = 0
                else: card_used = 1
                break

            # Record card used
            if resultant_move in cards[0].moves: card_used = 0
            else: card_used = 1
            break

        # Update boardstate by executing move
        execute_move(board, move, cards[card_used])
        
