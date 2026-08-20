import random

class RandomBot:
    def request_move(self, board):
        possible_moves = board.possible_moves()
        return random.choice(possible_moves)
