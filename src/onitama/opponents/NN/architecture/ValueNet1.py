import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------
# Card Encoder
# ------------------------

class CardEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16)  # no activation
        )

    def forward(self, cards):
        # cards: (B, 3, 16)
        x = cards.view(cards.size(0), -1)  # (B, 48)
        return self.net(x)  # (B, 16)


# ------------------------
# FiLM Residual Block
# ------------------------

class FiLMResBlock(nn.Module):
    def __init__(self, channels=16, card_dim=16):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        # FiLM generators
        self.gamma1 = nn.Linear(card_dim, channels)
        self.beta1 = nn.Linear(card_dim, channels)

        self.gamma2 = nn.Linear(card_dim, channels)
        self.beta2 = nn.Linear(card_dim, channels)

    def film(self, x, gamma, beta):
        # x: (B, C, H, W)
        # gamma, beta: (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

    def forward(self, x, card_embed):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        g1 = self.gamma1(card_embed)
        b1 = self.beta1(card_embed)
        out = self.film(out, g1, b1)

        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        g2 = self.gamma2(card_embed)
        b2 = self.beta2(card_embed)
        out = self.film(out, g2, b2)

        out = out + residual
        out = F.relu(out)

        return out


# ------------------------
# Full Network
# ------------------------

class ValueNet1(nn.Module):
    def __init__(self, num_blocks=6):
        super().__init__()

        self.card_encoder = CardEncoder()

        self.stem = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList(
            [FiLMResBlock(16, 16) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.Linear(16 + 16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)  

    def forward(self, board, cards):
        # board: (B, 4, 5, 5)
        # cards: (B, 3, 16)

        card_embed = self.card_encoder(cards)  # (B, 16)

        x = self.stem(board)

        for block in self.blocks:
            x = block(x, card_embed)

        # Global average pool
        x = x.mean(dim=[2, 3])  # (B, 16)

        x = torch.cat([x, card_embed], dim=1)  # (B, 32)

        value = self.head(x)

        return torch.tanh(value)