#pragma once
// mcts.h — Monte Carlo Tree Search with multi-threaded search
//
// Matches the Python MCTSBot logic:
//   - UCB selection (exploration constant)
//   - Expansion: copy board + apply move
//   - Rollout: heuristic evaluate + softmax OR uniform random
//   - Backpropagation: cumulative value, visits
//   - Multi-threaded with mutex on tree operations

#include "board.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace onitama {

// ------------------------------------------------------------------ //
//  Feature toggles (match Python behaviour when true)                  //
// ------------------------------------------------------------------ //
inline bool MCTS_PYTHON_COMPAT = true;  // force single-threaded

// ------------------------------------------------------------------ //
//  MCTS Node                                                          //
// ------------------------------------------------------------------ //

struct MCTSNodeCpp {
    BoardState board;
    MCTSNodeCpp* parent = nullptr;
    Move move;                              // move that led here
    std::vector<std::unique_ptr<MCTSNodeCpp>> children;
    std::vector<Move> untried_moves;
    int visits = 0;
    double value = 0.0;                     // cumulative reward (RED-positive)

    MCTSNodeCpp(const BoardState& b, MCTSNodeCpp* p = nullptr, Move m = Move())
        : board(b), parent(p), move(m)
    {
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

    bool is_fully_expanded() const { return untried_moves.empty(); }
    bool is_terminal() const { return board.is_terminal(); }
};

// ------------------------------------------------------------------ //
//  MCTS Bot                                                           //
// ------------------------------------------------------------------ //

class MCTSBot {
public:
    int num_threads;
    double exploration;
    double temperature;      // <= 0 means uniform random rollouts
    int time_limit_ms;
    int max_rollout_depth;

    mutable std::atomic<int> nodes_created{0};
    mutable std::atomic<int> rollouts_evaluated{0};

    MCTSBot(int time_ms = 1000, int threads = 4,
            double explore = 700.0, double temp = 0.7,
            int max_rollout = 200)
        : num_threads(threads), exploration(explore), temperature(temp),
          time_limit_ms(time_ms), max_rollout_depth(max_rollout) {}

    Move request_move(const BoardState& board) {
        auto root = std::make_unique<MCTSNodeCpp>(board);
        nodes_created.store(1, std::memory_order_relaxed);
        rollouts_evaluated.store(0, std::memory_order_relaxed);

        auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::milliseconds(time_limit_ms);

        std::mutex tree_mutex;

        auto worker = [&]() {
            while (std::chrono::steady_clock::now() < deadline) {
                MCTSNodeCpp* node;
                BoardState sim_board;
                {
                    std::lock_guard<std::mutex> lock(tree_mutex);
                    node = select(root.get());
                    if (!node->is_terminal() && !node->is_fully_expanded()) {
                        node = expand(node);
                    }
                    sim_board = node->board;
                }
                double reward = simulate(sim_board);
                {
                    std::lock_guard<std::mutex> lock(tree_mutex);
                    backpropagate(node, reward);
                }
            }
        };

        int n_threads = MCTS_PYTHON_COMPAT ? 1 : std::max(1, num_threads);
        std::vector<std::thread> threads;
        threads.reserve(n_threads);
        for (int i = 0; i < n_threads; ++i) threads.emplace_back(worker);
        for (auto& t : threads) t.join();

        // Select move with most visits
        MCTSNodeCpp* best = nullptr;
        int best_visits = -1;
        for (auto& child : root->children) {
            if (child->visits > best_visits) {
                best_visits = child->visits;
                best = child.get();
            }
        }

        if (best) return best->move;

        // Fallback
        auto moves = board.possible_moves_captures_first();
        return moves.empty() ? Move{} : moves[0];
    }

    int get_nodes_created() const { return nodes_created.load(); }
    int get_rollouts_evaluated() const { return rollouts_evaluated.load(); }

private:
    double ucb(const MCTSNodeCpp* node, double parent_visits, double mul) const {
        if (node->visits == 0) return std::numeric_limits<double>::infinity();
        double exploitation = mul * node->value / static_cast<double>(node->visits);
        double exploration_term = exploration *
            std::sqrt(std::log(std::max(1.0, parent_visits)) / node->visits);
        return exploitation + exploration_term;
    }

    MCTSNodeCpp* select(MCTSNodeCpp* node) const {
        while (!node->is_terminal() && node->is_fully_expanded() &&
               !node->children.empty()) {
            double pv = static_cast<double>(std::max(1, node->visits));
            double mul = node->board.is_red_turn() ? 1.0 : -1.0;
            node = (*std::max_element(
                node->children.begin(), node->children.end(),
                [&](const auto& a, const auto& b) {
                    return ucb(a.get(), pv, mul) < ucb(b.get(), pv, mul);
                })).get();
        }
        return node;
    }

    MCTSNodeCpp* expand(MCTSNodeCpp* node) {
        if (node->untried_moves.empty()) return node;
        Move m = node->untried_moves.back();
        node->untried_moves.pop_back();

        BoardState child_board = node->board;
        child_board.apply_move(m);
        node->children.push_back(
            std::make_unique<MCTSNodeCpp>(child_board, node, m));
        nodes_created.fetch_add(1, std::memory_order_relaxed);
        return node->children.back().get();
    }

    double simulate(BoardState sim) const {
        thread_local std::mt19937 rng(std::random_device{}());
        int depth = 0;

        while (!sim.is_terminal() && depth < max_rollout_depth) {
            auto moves = sim.possible_moves_captures_first();
            if (moves.empty()) break;

            Move selected;
            if (temperature <= 0.0) {
                // Uniform random
                std::uniform_int_distribution<int> dist(0, static_cast<int>(moves.size()) - 1);
                selected = moves[dist(rng)];
            } else {
                // Softmax-weighted by heuristic eval
                double mul = sim.is_red_turn() ? 1.0 : -1.0;
                std::vector<double> scores;
                scores.reserve(moves.size());
                for (const auto& m : moves) {
                    auto undo = sim.apply_move(m);
                    scores.push_back(mul * static_cast<double>(sim.evaluate()));
                    sim.undo_move(undo);
                }
                selected = softmax_select(moves, scores, temperature, rng);
            }

            sim.apply_move(selected);
            depth++;
        }

        rollouts_evaluated.fetch_add(1, std::memory_order_relaxed);
        return static_cast<double>(sim.evaluate(depth));
    }

    static Move softmax_select(const std::vector<Move>& moves,
                               const std::vector<double>& scores,
                               double temp, std::mt19937& rng) {
        if (moves.size() == 1) return moves[0];
        double max_score = *std::max_element(scores.begin(), scores.end());

        std::vector<double> exp_scores;
        exp_scores.reserve(moves.size());
        double sum_exp = 0.0;
        for (double s : scores) {
            double e = std::exp((s - max_score) / temp);
            exp_scores.push_back(e);
            sum_exp += e;
        }

        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(rng);
        double cumul = 0.0;
        for (size_t i = 0; i < moves.size(); ++i) {
            cumul += exp_scores[i] / sum_exp;
            if (r <= cumul) return moves[i];
        }
        return moves.back();
    }

    void backpropagate(MCTSNodeCpp* node, double reward) {
        while (node) {
            node->visits++;
            node->value += reward;
            node = node->parent;
        }
    }
};

}  // namespace onitama
