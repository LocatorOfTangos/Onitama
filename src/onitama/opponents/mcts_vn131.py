import torch
import random
import numpy as np
from copy import deepcopy
from onitama.opponents.NN.architecture.ValueNet1 import ValueNet1

try:
    from onitama.engine.onitama_engine import (
        BoardState as CppBoardState,
        ABSearchBot as CppABSearchBot,
        MCTSBot as CppMCTSBot,
        WinColour as CppWinColour,
        set_mcts_python_compat as cpp_set_mcts_python_compat,
        ValueLeafMCTSTree as CppValueLeafMCTSTree,
        from_python_board
    )
    _CPP_ENGINE = True
except ImportError:
    _CPP_ENGINE = False

class MCTS_VN131():
    def __init__(self, num_simulations = 2048): #2048
        self.num_simulations = num_simulations
        self.exploration = 1.4
        self.temperature = 0

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.model = ValueNet1().to(self.device)
        ckpt = torch.load("src/onitama/opponents/NN/final/vn1_3_1.pt", map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # JIT-trace for faster single-sample inference (~2x on CPU batch_size=1)
        try:
            _trace_b = torch.empty(1, 4, 5, 5)
            _trace_c = torch.empty(1, 3, 16)
            self.model = torch.jit.trace(self.model, (_trace_b, _trace_c))
            self.model.eval()
        except Exception:
            pass  # fall back to regular model if tracing fails

        
    def run_value_leaf_mcts(
        self,
        board,
        existing_tree=None,  # CppValueLeafMCTSTree to reuse (C++ path only)
    ):
        """Run value-leaf MCTS.

        Returns (root_proxy, root_value, tree) where:
        - root_proxy exposes .children with .move and .visits so _pick_root_move
            works unchanged.
        - tree is the live CppValueLeafMCTSTree (or None on the Python path).
            Pass it back as `existing_tree` on the next call after advancing it
            with tree.advance_to_child(move) to reuse all prior search work.
        """
        # ----------------------------------------------------------------
        # Fast path: C++ tree handles all select/expand/backprop overhead.
        # Python only runs the NN forward pass (one board at a time, CPU).
        # ----------------------------------------------------------------

        board = from_python_board(board) if _CPP_ENGINE else board

        if _CPP_ENGINE and _is_cpp_board(board):
            if existing_tree is not None:
                tree = existing_tree
            else:
                tree = CppValueLeafMCTSTree(board, self.exploration)
            board_buf  = torch.empty(1, 4, 5, 5)
            cards_buf  = torch.empty(1, 3, 16)

            # Hoist no_grad context and get numpy views once; avoids per-sim overhead
            with torch.no_grad():
                board_view = board_buf.numpy()
                cards_view = cards_buf.numpy()
                for _ in range(self.num_simulations):
                    node_idx, is_terminal, terminal_value, b_np, c_np = \
                        tree.select_and_expand()

                    if is_terminal:
                        value = terminal_value
                    else:
                        # Zero-copy: write directly into the tensor's backing memory
                        board_view[0] = b_np
                        cards_view[0] = c_np
                        value = float(self.model(board_buf, cards_buf).item())

                    tree.backpropagate(node_idx, value)

            root_value = tree.root_value()
            return _CppTreeRootProxy(tree), root_value, tree

        raise NotImplementedError()
        # # ----------------------------------------------------------------
        # # Pure-Python fallback
        # # ----------------------------------------------------------------
        # root = ValueLeafMCTSNode(board)

        # for _ in range(self.num_simulations):
        #     node = self._py_select_expand(root, self.exploration)

        #     if node.is_terminal():
        #         value = _board_terminal_value_current_player(node.board) or 0.0
        #     else:
        #         if not hasattr(node, "_encoded"):
        #             node._encoded = _encode_board(node.board)
        #         bt, ct = node._encoded
        #         with torch.no_grad():
        #             value = float(
        #                 self.model(bt.unsqueeze(0).to(device),
        #                     ct.unsqueeze(0).to(device)).item()
        #             )

        #     _py_backpropagate(node, value)

        # root_value = root.value_sum / root.visits if root.visits > 0 else 0.0
        # return root, root_value, None


    def _pick_root_move(self, root):
        """Select move proportional to visit counts (AlphaZero-style).
        temperature > 0: sample proportional to visits^(1/temperature)
        temperature <= 0: greedy (most visits)
        """
        if not root.children:
            return None
        if len(root.children) == 1:
            return root.children[0].move

        visits = [c.visits for c in root.children]

        if self.temperature <= 0:
            best_idx = max(range(len(root.children)), key=lambda i: visits[i])
            return root.children[best_idx].move

        # Exponentiated visit counts
        inv_temp = 1.0 / self.temperature
        weighted = [max(v, 0) ** inv_temp for v in visits]
        z = sum(weighted)
        if z == 0:
            return random.choice(root.children).move
        probs = [w / z for w in weighted]

        r = random.random()
        c = 0.0
        for i, p in enumerate(probs):
            c += p
            if r <= c:
                return root.children[i].move
        return root.children[-1].move


    def request_move(self, board, time_limit=None):

        root, root_value, tree = self.run_value_leaf_mcts(board=board)

        encoded_move = self._pick_root_move(root)

        return decode_move(encoded_move)
        

# ---------------------------------------------------------------------------
# Proxy so _pick_root_move works with either Python root or C++ tree result
# ---------------------------------------------------------------------------

class _ChildProxy:
    """Lightweight stand-in for ValueLeafMCTSNode children used by _pick_root_move."""
    __slots__ = ("move", "visits")

    def __init__(self, move, visits):
        self.move   = move
        self.visits = visits


class _CppTreeRootProxy:
    """Wraps CppValueLeafMCTSTree so _pick_root_move sees a .children list."""
    __slots__ = ("children",)

    def __init__(self, cpp_tree):
        self.children = [
            _ChildProxy(move, visits)
            for move, visits in cpp_tree.root_child_visits()
        ]

# Helpers

def _is_cpp_board(board):
    return _CPP_ENGINE and isinstance(board, CppBoardState)

def decode_move(encoded_move):
    """Convert engine/root move into Python Boardstate move format:
    ((from_x, from_y), (to_x, to_y), card_idx)
    """
    if encoded_move is None:
        return None

    # Already in target format
    if (
        isinstance(encoded_move, (tuple, list))
        and len(encoded_move) == 3
        and isinstance(encoded_move[0], (tuple, list))
        and isinstance(encoded_move[1], (tuple, list))
    ):
        (fx, fy), (tx, ty), card = encoded_move
        return ((int(fx), int(fy)), (int(tx), int(ty)), int(card))

    # C++ Move exposed directly (from_sq, to_sq, card_idx)
    if all(hasattr(encoded_move, a) for a in ("from_sq", "to_sq", "card_idx")):
        f = int(encoded_move.from_sq)
        t = int(encoded_move.to_sq)
        return ((f % 5, f // 5), (t % 5, t // 5), int(encoded_move.card_idx))

    # Move::to_coords() style object (fx, fy, tx, ty, card)
    if hasattr(encoded_move, "to_coords"):
        c = encoded_move.to_coords()
        if all(hasattr(c, a) for a in ("fx", "fy", "tx", "ty", "card")):
            return ((int(c.fx), int(c.fy)), (int(c.tx), int(c.ty)), int(c.card))

        # tuple/list fallback from to_coords
        if isinstance(c, (tuple, list)) and len(c) == 5:
            fx, fy, tx, ty, card = c
            return ((int(fx), int(fy)), (int(tx), int(ty)), int(card))

    # Flat tuple/list fallback: (fx, fy, tx, ty, card)
    if isinstance(encoded_move, (tuple, list)) and len(encoded_move) == 5:
        fx, fy, tx, ty, card = encoded_move
        return ((int(fx), int(fy)), (int(tx), int(ty)), int(card))

    raise TypeError(f"Unsupported move format: {encoded_move!r}")