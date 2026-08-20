import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NeuralHeuristicNet(nn.Module):
    def __init__(self, num_game_states = 5):
        super().__init__()

        self.game_state = nn.Sequential(
            nn.Linear(100, num_game_states),
            nn.ReLU()
        )

        self.pos_values = nn.Linear(num_game_states, 100)


        self.cos = nn.CosineSimilarity(dim=1)


    # def dot(self, X, Y):
    #     B, d = X.shape
    #     X = X.reshape(B, 1, d)
    #     Y = Y.reshape(B, d, 1)
    #     return torch.matmul(X, Y).squeeze(1)


    def forward(self, board, cards):
        # board: (B, 4, 5, 5)

        x = torch.flatten(board, start_dim=1)  # (B, 100)

        # Game state should be a linear function of the positions
        # ReLU enforces non-negativity. Consider using a max to
        # simply 'select' the game state.
        game_state = self.game_state(x)  # (B, num_game_states)

        # The game state determines a value for each (piece,
        # position) pair. 
        pos_values = self.pos_values(game_state)  # (B, 100)

        # Dot product applies the prescribed (piece, position)
        # value for each living piece. Magnitude scaling handles
        # values blowing up earlier, and ensures output between
        # [-1, 1]. Both are handled by vector cosine similarity.
        value = self.cos(x, pos_values).unsqueeze(1)  # (B, 1)

        return value