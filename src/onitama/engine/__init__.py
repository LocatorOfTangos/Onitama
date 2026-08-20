"""C++ game engine for Onitama — bitboard board state, minimax, AB search, MCTS."""

try:
    from .onitama_engine import (  # noqa: F401
        BoardState,
        Move,
        MinimaxTree,
        ABSearchBot,
        MCTSBot,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
