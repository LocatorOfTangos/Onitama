from back import *

def request_bot_move(board):

    possible_moves = board.possible_moves()
    
    return random.sample(possible_moves,1)[0]