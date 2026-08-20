import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


# ----------------------------------
# Dynamic Convolution Residual Block
# ----------------------------------

class DynamicConvResBlock(nn.Module):
    """Residual block where conv weights are generated from the card embedding."""

    def __init__(self, channels=16, card_dim=16, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.kernel_scale = 1.0 / math.sqrt(channels * kernel_size * kernel_size)

        weight_size = channels * channels * kernel_size * kernel_size

        # Weight and bias generators for conv1
        self.weight_gen1 = nn.Linear(card_dim, weight_size)
        self.bias_gen1 = nn.Linear(card_dim, channels)
        self.bn1 = nn.BatchNorm2d(channels)

        # Weight and bias generators for conv2
        self.weight_gen2 = nn.Linear(card_dim, weight_size)
        self.bias_gen2 = nn.Linear(card_dim, channels)
        self.bn2 = nn.BatchNorm2d(channels)

        # Small init so residual blocks start near-identity
        for gen in [self.weight_gen1, self.weight_gen2]:
            nn.init.normal_(gen.weight, std=0.01)
            nn.init.zeros_(gen.bias)
        for gen in [self.bias_gen1, self.bias_gen2]:
            nn.init.zeros_(gen.weight)
            nn.init.zeros_(gen.bias)

    def dynamic_conv(self, x, weight_gen, bias_gen, card_embed):
        B, C, H, W = x.shape
        input_dtype = x.dtype

        # Generate per-sample conv weights and biases from card embedding
        weights = torch.tanh(weight_gen(card_embed)) * self.kernel_scale
        weights = weights.view(B * self.channels, self.channels,
                               self.kernel_size, self.kernel_size)

        biases = torch.tanh(bias_gen(card_embed))
        biases = biases.view(B * self.channels)

        # Grouped conv trick: treat batch dim as groups for per-sample filters
        x = x.reshape(1, B * C, H, W).float()
        weights = weights.float()
        biases = biases.float()

        out = F.conv2d(x, weights, biases, padding=self.padding, groups=B)
        out = out.view(B, self.channels, H, W)
        return out.to(dtype=input_dtype)

    def forward(self, x, card_embed):
        residual = x

        out = self.dynamic_conv(x, self.weight_gen1, self.bias_gen1, card_embed)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.dynamic_conv(out, self.weight_gen2, self.bias_gen2, card_embed)
        out = self.bn2(out)

        out = out + residual
        out = F.relu(out)

        return out


# ------------------------
# Full Network
# ------------------------

class ValueNet0(nn.Module):
    def __init__(self, num_blocks=6):
        super().__init__()

        self.card_encoder = CardEncoder()

        self.stem = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList(
            [DynamicConvResBlock(16, 16) for _ in range(num_blocks)]
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