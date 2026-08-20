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
    
class RuleUpdate(nn.Module):
    def __init__(self, latent_moves=16, card_dim=16):
        super().__init__()

        self.linear1 = nn.Linear(latent_moves+2*card_dim, 32)
        self.linear2 = nn.Linear(32, 16)

    def forward(self, x, card_embed, prime_card_embed):
        out = torch.cat([x, card_embed, prime_card_embed], dim=1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)

        return out

class RuleUpdateBlock(nn.Module):
    def __init__(self, rule_update, channels=16, latent_moves=16, card_dim=16):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, latent_moves, 3, padding=1)
        # self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, latent_moves, 3, padding=1)
        # self.bn2 = nn.BatchNorm2d(channels)

        # Problematic - should be a hadamard product over dimension 1, not a cosine similarity, but torch does not support it.
        self.cos = nn.CosineSimilarity(dim=1) #(B, 16, 5, 5) -> (B, 16)

        self.rule_update = rule_update

    def forward(self, x0, x1, card_embed, prime_card_embed):

        out0 = self.conv1(x0)
        out1 = self.conv2(x1)

        out = self.cos(out0, out1)
        out = self.rule_update(out, card_embed, prime_card_embed)

        return x1, out


# ------------------------
# Full Network
# ------------------------

class ValueNet2(nn.Module):
    def __init__(self, num_blocks=6):
        super().__init__()

        self.num_blocks = num_blocks

        self.card_encoder = CardEncoder()
        self.rule_update = RuleUpdate()

        self.stem = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.FiLMResBlocks = nn.ModuleList(
            [FiLMResBlock(16, 16) for _ in range(num_blocks)]
        )

        self.RuleUpdateBlocks = nn.ModuleList(
            [RuleUpdateBlock(self.rule_update, 16, 16, 16) for _ in range(num_blocks - 1)]
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
        prime_card_embed = card_embed

        x0 = self.stem(board)

        for i in range(self.num_blocks - 1):
            x1 = self.FiLMResBlocks[i](x0, card_embed)
            x0, card_embed = self.RuleUpdateBlocks[i](x0, x1, card_embed, prime_card_embed)
        
        x = self.FiLMResBlocks[-1](x0, card_embed)

        # Global average pool
        x = x.mean(dim=[2, 3])  # (B, 16)

        x = torch.cat([x, card_embed], dim=1)  # (B, 32)

        value = self.head(x)

        return torch.tanh(value)