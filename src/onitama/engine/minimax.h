#pragma once
// minimax.h — Full minimax tree with batched NN leaf evaluation interface
//
// Usage (from Python):
//   tree = MinimaxTree.build(board_state, max_depth)
//   leaf_boards, leaf_cards = tree.get_leaf_tensors()
//   # ... run NN on GPU ...
//   tree.set_leaf_values(values_array)
//   tree.backup()
//   examples = tree.get_training_examples(example_depth_cutoff)
//   move = tree.best_move()  or  tree.select_move_temperature(0.15)

#include "board.h"
#include "encode.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace onitama {

// ------------------------------------------------------------------ //
//  Arena-allocated minimax tree                                       //
// ------------------------------------------------------------------ //

struct MinimaxNode {
    BoardEncoding encoding;      // for NN training examples
    float value = 0.0f;          // backed-up or leaf value
    int16_t depth = 0;
    bool terminal = false;
    bool is_leaf  = false;       // needs NN eval (non-terminal, depth == max_depth)

    // Children: explicit index list (DFS expansion means children are
    // NOT at consecutive indices in the arena).
    std::vector<int> child_indices;

    // Move that led to this node (only meaningful for non-root)
    Move move_from_parent;
};

class MinimaxTree {
public:
    std::vector<MinimaxNode> nodes;
    std::vector<int> leaf_indices;       // indices of nodes needing NN eval
    int max_depth = 0;

    // ---- Phase 1: Build ---- //

    static MinimaxTree build(const BoardState& root, int max_depth) {
        MinimaxTree tree;
        tree.max_depth = max_depth;
        tree.nodes.reserve(8192);  // reasonable initial capacity

        // Build root
        tree.expand(root, 0);

        return tree;
    }

    // ---- Phase 2a: Get leaf tensors ---- //

    struct LeafTensors {
        std::vector<float> boards;  // N * 4 * 5 * 5
        std::vector<float> cards;   // N * 3 * 16
        int count;
    };

    LeafTensors get_leaf_tensors() const {
        int n = static_cast<int>(leaf_indices.size());
        LeafTensors lt;
        lt.count = n;
        lt.boards.resize(n * 4 * 5 * 5);
        lt.cards.resize(n * 3 * 16);

        for (int i = 0; i < n; ++i) {
            const auto& enc = nodes[leaf_indices[i]].encoding;
            std::memcpy(&lt.boards[i * 100], &enc.board, 100 * sizeof(float));
            std::memcpy(&lt.cards[i * 48],   &enc.cards,  48 * sizeof(float));
        }
        return lt;
    }

    // ---- Phase 2b: Set leaf values ---- //

    void set_leaf_values(const float* values, int count) {
        for (int i = 0; i < count && i < static_cast<int>(leaf_indices.size()); ++i) {
            nodes[leaf_indices[i]].value = values[i];
        }
    }

    // ---- Phase 3: Negamax backup ---- //

    void backup() {
        backup_node(0);
    }

    // ---- Phase 4: Extract results ---- //

    Move best_move() const {
        if (nodes.empty() || nodes[0].child_indices.empty()) return {};
        const auto& root = nodes[0];
        int best_idx = -1;
        float best_val = -1e30f;
        for (int ci : root.child_indices) {
            float v = -nodes[ci].value;
            if (v > best_val) {
                best_val = v;
                best_idx = ci;
            }
        }
        return (best_idx >= 0) ? nodes[best_idx].move_from_parent : Move{};
    }

    Move select_move_temperature(double temperature, std::mt19937& rng) const {
        if (nodes.empty() || nodes[0].child_indices.empty()) return {};
        const auto& root = nodes[0];
        int nc = static_cast<int>(root.child_indices.size());
        if (nc == 1) return nodes[root.child_indices[0]].move_from_parent;

        // Gather child values (negated for parent perspective)
        std::vector<double> vals(nc);
        double max_v = -1e30;
        for (int i = 0; i < nc; ++i) {
            vals[i] = -static_cast<double>(nodes[root.child_indices[i]].value);
            if (vals[i] > max_v) max_v = vals[i];
        }

        // Numerically stable softmax
        std::vector<double> probs(nc);
        double sum = 0.0;
        for (int i = 0; i < nc; ++i) {
            probs[i] = std::exp((vals[i] - max_v) / temperature);
            sum += probs[i];
        }
        for (auto& p : probs) p /= sum;

        // Sample
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(rng);
        double cumul = 0.0;
        for (int i = 0; i < nc; ++i) {
            cumul += probs[i];
            if (r <= cumul) return nodes[root.child_indices[i]].move_from_parent;
        }
        return nodes[root.child_indices.back()].move_from_parent;
    }

    // Training examples: (board_encoding, value) for nodes with depth < example_depth
    struct TrainingExample {
        BoardEncoding encoding;
        float value;
    };

    std::vector<TrainingExample> get_training_examples(int example_depth) const {
        std::vector<TrainingExample> examples;
        examples.reserve(nodes.size());
        collect_examples(0, example_depth, examples);
        return examples;
    }

private:
    // Recursive tree builder
    int expand(const BoardState& board, int depth) {
        int idx = static_cast<int>(nodes.size());
        nodes.emplace_back();
        // Don't hold a reference — vector may reallocate during child expansion.
        nodes[idx].depth = static_cast<int16_t>(depth);
        nodes[idx].encoding = encode_for_nn(board);

        // Terminal?
        auto w = board.is_won();
        if (w) {
            nodes[idx].terminal = true;
            bool red_turn = board.is_red_turn();
            nodes[idx].value = (w.colour == WinColour::RED) == red_turn ? 1.0f : -1.0f;
            return idx;
        }

        // Leaf (max depth)?
        if (depth >= max_depth) {
            nodes[idx].is_leaf = true;
            leaf_indices.push_back(idx);
            return idx;
        }

        // Expand children
        auto moves = board.possible_moves();
        std::vector<int> children;
        children.reserve(moves.size());

        BoardState child_board;
        for (const auto& m : moves) {
            child_board = board;
            child_board.apply_move(m);
            int child_idx = expand(child_board, depth + 1);
            nodes[child_idx].move_from_parent = m;
            children.push_back(child_idx);
        }

        // Store child indices (nodes[idx] is stable now — no more expansions)
        nodes[idx].child_indices = std::move(children);

        return idx;
    }

    void backup_node(int idx) {
        auto& node = nodes[idx];
        if (node.terminal || node.is_leaf) return;
        if (node.child_indices.empty()) return;

        float best = -1e30f;
        for (int ci : node.child_indices) {
            backup_node(ci);
            float v = -nodes[ci].value;
            if (v > best) best = v;
        }
        node.value = best;
    }

    void collect_examples(int idx, int example_depth,
                          std::vector<TrainingExample>& out) const {
        const auto& node = nodes[idx];
        if (node.depth < example_depth) {
            out.push_back({node.encoding, node.value});
        }
        for (int ci : node.child_indices) {
            collect_examples(ci, example_depth, out);
        }
    }
};

}  // namespace onitama
