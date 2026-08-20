import copy
import random

class ShortSightedBot:
    def request_move(self, board):
        possible_moves = board.possible_moves()

        scored = []
        for pos_idx, move in enumerate(possible_moves):
            simulation_board = board.copy()
            simulation_board.execute_move(move)
            possible_countermoves = simulation_board.possible_moves()
            worst_outcome = 300
            for cm in possible_countermoves:
                simsim = simulation_board.copy()
                simsim.execute_move(cm)
                evaluation = simsim.evaluate_position()
                if evaluation < worst_outcome:
                    worst_outcome = evaluation
            scored.append((move, worst_outcome))

        best_score = max(score for (_m, score) in scored)
        best_moves = [m for (m, s) in scored if s == best_score]
        return random.choice(best_moves)
