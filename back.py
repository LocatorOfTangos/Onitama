from front import *
import random
import re

INITIAL_POSITIONS = [(2,4),(2,0),(0,4),(1,4),(3,4),(4,4),(0,0),(1,0),(3,0),(4,0)]

class CARD:
    def __init__(self, name, stamp, moves):
        self.name = name
        self.stamp = stamp
        self.moves = moves

DECK = [
    CARD('tiger', 'red', [(0,2), (0,-1)]),
    CARD('monkey', 'red', [(1,1),(-1,1),(-1,-1),(1,-1)]),
    CARD('dragon', 'red', [(2,1),(-2,1),(-1,-1),(1,-1)]),
    CARD('crab', 'blue', [(2,0),(-2,0),(0,1)]),
    CARD('mantis', 'blue', [(1,1),(-1,1),(0,-1)])
]

class BOARD:
    def __init__(self, turn, positions, cards)
        self.turn = turn
        self.positions = positions
        self.cards = cards

def initialise_board():
    sel = []
    while len(sel)<5:
        num = random.randint(0,len(DECK)-1)
        if num not in sel: sel.append(num)
    cards = [DECK[x] for x in sel]

    turn = 0 if cards[4].stamp == "RED" else turn = 1

    return BOARD(turn, INITIAL_POSITIONS, cards)

def print_board():
    pass #TODO

def print_victory():
    pass #TODO

def execute_move(move):
    pass #TODO

def is_won(board):
    if board.positions[0] == (2,0): return ("RED", "WIND")
    elif board.positions[1] == (2,4): return ("BLUE", "WIND")
    elif board.positions[0] == (-1,-1): return ("BLUE", "ROCK")
    elif board.positions[1] == (-1,-1): return ("RED", "ROCK")
    else return NULL

def two_player_mode():
    board = initialise_board()
    while TRUE:

        print_board(board)

        if board.is_won() is not NULL:
            print_victory(board.iswon())
            return

        if board.turn % 2 == 0:
            print("RED's turn.")
        else: print("BLUE's turn.")

        while TRUE:
            move_input = input("Move a piece by typing it's starting and ending coords: \"(x0,y0),(x1,y1)\".")
            if not re.match("(\d,\d),(\d,\d)"):
                print("Could not parse input.")
                continue
            ints = re.findall("\d", move_input)
            move = [ (ints[0], ints[1]), (ints[2], ints[3]) ]
            
            #TODO: reject moves not moving player's own piece
            #TODO: reject moves not landing on an empty square or enemy piece

            break

        execute_move(move)
        
