#pragma once
// value_mcts.h — ValueLeaf MCTS tree for NN-guided self-play
//
// Design goals:
//   - Flat node arena (no unique_ptr / heap allocation per node)
//   - Sequential or batched select → expand → NN-eval → backprop
//   - Optional virtual loss for batched leaf evaluation (train4 style)
//   - All tree logic stays in C++; Python only calls forward pass for each leaf
//   - select_and_expand() returns a SelectResult with the encoded board so Python
//     can run the NN without touching any C++ board objects
//   - backpropagate(node_idx, value) closes the loop
//
// UCB formula matches VN1_train3.py exactly:
//   exploitation = -(child.value_sum / child.visits)   // convert to parent POV
//   exploration  = C * sqrt(log(parent_visits) / child.visits)
//
// Node value convention (same as Python):
//   value_sum at a node is accumulated from that node's CURRENT PLAYER's perspective.
//   Backpropagation alternates sign at each level.

#include "board.h"
#include "encode.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace onitama {

// ------------------------------------------------------------------ //
//  Internal node type                                                 //
// ------------------------------------------------------------------ //

struct VLNode {
    BoardState board;
    int         parent_idx  = -1;
    Move        move;                   // move that led here from parent
    std::vector<int>  child_indices;
    std::vector<Move> untried_moves;
    int   visits    = 0;
    float value_sum = 0.0f;
    int   vl_count  = 0;  // outstanding virtual-loss traversals through this node

    VLNode() = default;
    VLNode(const BoardState& b, int parent, Move m)
        : board(b), parent_idx(parent), move(m)
    {
        // Populate untried moves — instant-win short-circuit, captures first
        auto sep = board.possible_moves_search_optimised();
        if (sep.has_instant_win) {
            untried_moves = {sep.winning_move};
        } else {
            untried_moves.reserve(sep.captures.size() + sep.other.size());
            untried_moves.insert(untried_moves.end(),
                                 sep.captures.begin(), sep.captures.end());
            untried_moves.insert(untried_moves.end(),
                                 sep.other.begin(), sep.other.end());
        }
    }

    bool is_terminal()      const { return board.is_terminal(); }
    bool is_fully_expanded() const { return untried_moves.empty(); }

    float mean_value() const {
        return visits > 0 ? value_sum / static_cast<float>(visits) : 0.0f;
    }
};

// ------------------------------------------------------------------ //
//  Return type for select_and_expand()                               //
// ------------------------------------------------------------------ //

struct VLSelectResult {
    int   node_idx;          // index into tree.nodes
    bool  is_terminal;
    float terminal_value;    // from current player's POV; only valid if is_terminal
    BoardEncoding encoding;  // NN input; only valid if !is_terminal
};

// ------------------------------------------------------------------ //
//  ValueLeafMCTSTree                                                 //
// ------------------------------------------------------------------ //

class ValueLeafMCTSTree {
public:
    std::vector<VLNode> nodes;
    double exploration;
    float  virtual_loss_magnitude = 0.0f;  // >0 enables virtual loss
    int root_idx = 0;   // index of the current root (advances on tree reuse)

    explicit ValueLeafMCTSTree(const BoardState& root_board,
                                double exploration_c = 1.4,
                                float virtual_loss = 0.0f)
        : exploration(exploration_c), virtual_loss_magnitude(virtual_loss)
    {
        // Pre-allocate; each simulation creates at most one new node
        nodes.reserve(2048);
        nodes.emplace_back(root_board, /*parent*/ -1, Move{});
    }

    // -------------------------------------------------------------- //
    //  Tree-reuse interface                                           //
    //                                                                  //
    //  After selecting a move, call advance_to_child(m) to re-root    //
    //  the tree at the chosen child.  The child's entire subtree is   //
    //  retained and its visit statistics carry forward into the next  //
    //  MCTS run, saving the work that was already done for that        //
    //  branch.  Returns false (and leaves the tree unchanged) when    //
    //  the move was never explored, so callers know to fall back to a //
    //  fresh tree instead.                                            //
    // -------------------------------------------------------------- //
    bool advance_to_child(const Move& m) {
        for (int ci : nodes[root_idx].child_indices) {
            if (nodes[ci].move == m) {
                root_idx = ci;
                // Detach new root from its old parent so backpropagate()
                // stops here naturally (it walks parent_idx until -1).
                nodes[root_idx].parent_idx = -1;
                // Compact the arena: discard all nodes that are no longer
                // reachable from the new root (old root, its other children
                // and their entire subtrees).  Without this the vector grows
                // monotonically for the lifetime of the game.
                compact();
                return true;
            }
        }
        return false;  // move was never expanded; caller should create a fresh tree
    }

    // Discard all nodes and start over from a new board position.
    // Use this when advance_to_child() returns false.
    void reset_to_board(const BoardState& board) {
        nodes.clear();
        nodes.reserve(2048);
        nodes.emplace_back(board, /*parent*/ -1, Move{});
        root_idx = 0;
    }

    // -------------------------------------------------------------- //
    //  select_and_expand                                              //
    //                                                                  //
    //  Walks down the tree via UCB, expands if possible, returns      //
    //  the selected leaf index plus its encoded board for NN eval.    //
    // -------------------------------------------------------------- //
    VLSelectResult select_and_expand() {
        int idx = root_idx;

        // Selection: descend while fully expanded, not terminal, has children
        while (!nodes[idx].is_terminal()
               && nodes[idx].is_fully_expanded()
               && !nodes[idx].child_indices.empty())
        {
            idx = best_ucb_child(idx);
        }

        // Expansion: create one new child if possible
        if (!nodes[idx].is_terminal() && !nodes[idx].is_fully_expanded()) {
            // Pop last untried move (captures are inserted first, so
            // untried_moves is [captures..., other...] and we pop from back
            // which gives other/non-captures first; reasonable ordering)
            Move m = nodes[idx].untried_moves.back();
            nodes[idx].untried_moves.pop_back();

            BoardState child_board = nodes[idx].board;
            child_board.apply_move(m);

            int child_idx = static_cast<int>(nodes.size());
            nodes[idx].child_indices.push_back(child_idx);
            nodes.emplace_back(child_board, idx, m);
            idx = child_idx;
        }

        // Build result
        VLSelectResult res;
        res.node_idx = idx;

        if (nodes[idx].is_terminal()) {
            res.is_terminal    = true;
            res.terminal_value = terminal_value_for_current_player(nodes[idx].board);
            // zero out encoding to avoid uninitialized reads
            std::memset(&res.encoding, 0, sizeof(res.encoding));
        } else {
            res.is_terminal    = false;
            res.terminal_value = 0.0f;
            res.encoding       = encode_for_nn(nodes[idx].board);
        }

        return res;
    }

    // -------------------------------------------------------------- //
    //  backpropagate                                                  //
    //                                                                  //
    //  Walk from node_idx to root, alternating sign.                  //
    //  value is from node_idx's current player's perspective.         //
    // -------------------------------------------------------------- //
    void backpropagate(int node_idx, float value) {
        int cur = node_idx;
        float v  = value;
        while (cur >= 0) {
            nodes[cur].visits++;
            nodes[cur].value_sum += v;
            v   = -v;
            cur = nodes[cur].parent_idx;
        }
    }

    // -------------------------------------------------------------- //
    //  Virtual-loss helpers                                           //
    //                                                                  //
    //  Call apply_virtual_loss(idx) right after select_and_expand()   //
    //  for a non-terminal leaf, before the NN evaluation starts.      //
    //  This increments vl_count along the path so that subsequent     //
    //  selections in the same batch see the node as provisionally      //
    //  less attractive (UCB uses effective_visits/effective_value).   //
    //                                                                  //
    //  Call undo_virtual_loss(idx) just before backpropagate(), once  //
    //  the real NN value is known.  This decrements vl_count so the   //
    //  provisional effect is cleanly replaced by the real value.      //
    // -------------------------------------------------------------- //
    void apply_virtual_loss(int node_idx) {
        int cur = node_idx;
        while (cur >= 0) {
            nodes[cur].vl_count++;
            cur = nodes[cur].parent_idx;
        }
    }

    void undo_virtual_loss(int node_idx) {
        int cur = node_idx;
        while (cur >= 0) {
            if (nodes[cur].vl_count > 0)
                nodes[cur].vl_count--;
            cur = nodes[cur].parent_idx;
        }
    }

    // -------------------------------------------------------------- //
    //  Query helpers                                                  //
    // -------------------------------------------------------------- //

    float root_value() const {
        const auto& root = nodes[root_idx];
        return root.visits > 0 ? root.value_sum / static_cast<float>(root.visits)
                                : 0.0f;
    }

    // Returns (Move, visits) for each direct child of root
    std::vector<std::pair<Move, int>> root_child_visits() const {
        std::vector<std::pair<Move, int>> out;
        out.reserve(nodes[root_idx].child_indices.size());
        for (int ci : nodes[root_idx].child_indices) {
            out.emplace_back(nodes[ci].move, nodes[ci].visits);
        }
        return out;
    }

    // Greedy best move (most visits)
    Move best_move_greedy() const {
        const auto& root = nodes[root_idx];
        int best_ci = -1, best_v = -1;
        for (int ci : root.child_indices) {
            if (nodes[ci].visits > best_v) {
                best_v  = nodes[ci].visits;
                best_ci = ci;
            }
        }
        if (best_ci < 0) {
            // No children explored — fall back to first legal move
            auto moves = root.board.possible_moves_captures_first();
            return moves.empty() ? Move{} : moves[0];
        }
        return nodes[best_ci].move;
    }

    // Temperature-weighted best move (AlphaZero style)
    // temperature <= 0 → greedy
    Move best_move_temperature(float temperature, uint32_t rng_seed = 0) const {
        if (temperature <= 0.0f) return best_move_greedy();

        const auto& root = nodes[root_idx];
        if (root.child_indices.empty()) return best_move_greedy();

        float inv_temp = 1.0f / temperature;
        std::vector<float> weighted;
        weighted.reserve(root.child_indices.size());
        float z = 0.0f;
        for (int ci : root.child_indices) {
            float w = std::pow(static_cast<float>(std::max(nodes[ci].visits, 0)), inv_temp);
            weighted.push_back(w);
            z += w;
        }
        if (z == 0.0f) return best_move_greedy();

        // Simple LCG for reproducible sampling without requiring <random> state
        uint32_t rng = rng_seed ^ 0xdeadbeef;
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        float r = static_cast<float>(rng) / static_cast<float>(0xffffffffu);

        float cumul = 0.0f;
        for (size_t i = 0; i < weighted.size(); ++i) {
            cumul += weighted[i] / z;
            if (r <= cumul) return nodes[root.child_indices[i]].move;
        }
        return nodes[root.child_indices.back()].move;
    }

private:
    // -------------------------------------------------------------- //
    //  Arena compaction                                               //
    //                                                                  //
    //  Called by advance_to_child() after re-rooting.  Performs a    //
    //  DFS from root_idx, collects all reachable nodes, remaps every  //
    //  parent/child index, and rebuilds the nodes vector in-place.   //
    //  After compaction root_idx is always 0 and vector capacity is  //
    //  trimmed to (live nodes + 2048) so arena size stays bounded.   //
    // -------------------------------------------------------------- //
    void compact() {
        // DFS to enumerate reachable indices (tree has no cycles)
        std::vector<int> reachable;
        reachable.reserve(nodes.size());
        std::vector<int> stack = {root_idx};
        std::vector<bool> visited(nodes.size(), false);
        visited[root_idx] = true;
        while (!stack.empty()) {
            int cur = stack.back(); stack.pop_back();
            reachable.push_back(cur);
            for (int ci : nodes[cur].child_indices) {
                if (!visited[ci]) {
                    visited[ci] = true;
                    stack.push_back(ci);
                }
            }
        }

        // Build old-index → new-index remap table
        std::vector<int> remap(nodes.size(), -1);
        for (int new_i = 0; new_i < static_cast<int>(reachable.size()); ++new_i)
            remap[reachable[new_i]] = new_i;

        // Rebuild arena with only live nodes, all indices remapped
        std::vector<VLNode> compacted;
        compacted.reserve(reachable.size() + 2048);  // headroom for next search
        for (int old_i : reachable) {
            VLNode n = std::move(nodes[old_i]);
            if (n.parent_idx >= 0)
                n.parent_idx = remap[n.parent_idx];
            for (int& ci : n.child_indices)
                ci = remap[ci];
            compacted.push_back(std::move(n));
        }

        nodes  = std::move(compacted);
        root_idx = 0;  // root is always index 0 after compaction
    }

    // -------------------------------------------------------------- //
    //  UCB child selection (matching Python VN1_train3 formula)       //
    // -------------------------------------------------------------- //
    int best_ucb_child(int parent_idx) const {
        const VLNode& parent = nodes[parent_idx];
        double pv = static_cast<double>(std::max(1, parent.visits));

        int    best_ci  = parent.child_indices[0];
        double best_ucb = -std::numeric_limits<double>::infinity();

        for (int ci : parent.child_indices) {
            const VLNode& child = nodes[ci];
            // Effective stats: pending virtual-loss traversals make the node
            // look provisionally worse (higher value_sum = worse for parent
            // in exploitation = -value_sum/visits).
            int   eff_visits = child.visits + child.vl_count;
            float eff_vsum   = child.value_sum +
                               static_cast<float>(child.vl_count) * virtual_loss_magnitude;
            double ucb_val;
            if (eff_visits == 0) {
                ucb_val = std::numeric_limits<double>::infinity();
            } else {
                double exploitation = -(static_cast<double>(eff_vsum) /
                                        static_cast<double>(eff_visits));
                double expl_term    = exploration *
                    std::sqrt(std::log(pv) / static_cast<double>(eff_visits));
                ucb_val = exploitation + expl_term;
            }
            if (ucb_val > best_ucb) {
                best_ucb = ucb_val;
                best_ci  = ci;
            }
        }
        return best_ci;
    }

    // Terminal value from the current player's perspective
    static float terminal_value_for_current_player(const BoardState& board) {
        const auto win = board.is_won();
        if (!win) return 0.0f;
        bool current_is_red = board.is_red_turn();
        if (win.colour == WinColour::RED) {
            return current_is_red ? 1.0f : -1.0f;
        } else {
            return current_is_red ? -1.0f : 1.0f;
        }
    }
};

}  // namespace onitama
