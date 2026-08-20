"""
Python wrapper for the C++ parallel MCTS bot.
Compiles the C++ source and invokes it via subprocess for actual move generation.
"""
import subprocess
from pathlib import Path

OPT_TEMP = 38.0 # 0.7
OPT_EXPLORE = 700.0 # 1400.0 has beaten ab search, but often causes immediate aggressive moves that lose the game.

class PMCTSBot:
    def __init__(self, time_limit_ms=1000, num_threads=8, temperature=OPT_TEMP, exploration=OPT_EXPLORE):
        self.time_limit_ms = time_limit_ms
        self.num_threads = num_threads
        self.temperature = temperature
        self.exploration = exploration
        self.cpp_source = Path(__file__).parent / "pmcts_bot.cpp"
        self.cpp_binary = Path(__file__).parent / "pmcts_bot"
        self._ensure_compiled()
    
    def _ensure_compiled(self):
        """Compile C++ source if binary doesn't exist or source is newer."""
        if not self.cpp_binary.exists() or \
           self.cpp_source.stat().st_mtime > self.cpp_binary.stat().st_mtime:
            self._compile()
    
    def _compile(self):
        """Compile the C++ MCTS bot."""
        compile_cmd = [
            "g++",
            "-std=c++17",
            "-O3",
            "-pthread",
            str(self.cpp_source),
            "-o",
            str(self.cpp_binary)
        ]
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"C++ compilation failed:\n{result.stderr}")
    
    def _serialize_board(self, board, time_ms=None):
        """Convert Python Boardstate to C++ input format."""
        # Use provided time or default
        if time_ms is None:
            time_ms = self.time_limit_ms
            
        # Positions: 10 space-separated x,y pairs (use -1,-1 for None)
        parts = []
        for pos in board.positions:
            if pos is None:
                parts.extend(["-1", "-1"])
            else:
                parts.extend([str(pos[0]), str(pos[1])])
        
        # Red cards (2 indices)
        parts.append(str(board.red_cards[0].idx))
        parts.append(str(board.red_cards[1].idx))
        
        # Blue cards (2 indices)
        parts.append(str(board.blue_cards[0].idx))
        parts.append(str(board.blue_cards[1].idx))
        
        # Trans card (1 index)
        parts.append(str(board.trans_card.idx))
        
        # Turn num
        parts.append(str(board.turn_num))
        
        # Red start (0 or 1)
        parts.append("1" if board.red_start else "0")
        
        # Time limit and thread count
        parts.append(str(time_ms))
        parts.append(str(self.num_threads))
        
        # Exploration and temperature
        parts.append(str(self.exploration))
        # temperature <= 0 means uniform random rollouts in C++
        parts.append(str(self.temperature) if self.temperature is not None else "-1")
        
        return " ".join(parts)
    
    def _deserialize_move(self, output):
        """Parse C++ output to Python move format."""
        parts = output.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Invalid C++ output: {output}")
        
        start_x, start_y, dest_x, dest_y, card_idx = map(int, parts)
        return ((start_x, start_y), (dest_x, dest_y), card_idx)
    
    def request_move(self, board, time_limit=None):
        """Request a move from the C++ bot."""
        # Use provided time_limit or default from __init__
        time_ms = int(time_limit * 1000) if time_limit is not None else self.time_limit_ms
        
        input_data = self._serialize_board(board, time_ms)
        
        result = subprocess.run(
            [str(self.cpp_binary)],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=time_ms / 1000 + 2  # Add 2s buffer for overhead
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"C++ bot execution failed:\n{result.stderr}")
        
        return self._deserialize_move(result.stdout)
