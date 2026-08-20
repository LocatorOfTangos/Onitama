#pragma once
// absearch.h — Alpha-beta search with move ordering and mobility bonus
//
// When toggles are all off, matches ab_search_bot.py exactly:
//   - depth counts UP from 0 (root) to max_depth
//   - possible_moves_search_optimised() returns captures-first; instant
//     wins are single-element lists that flow through simulate() normally
//   - Mobility bonus: at non-root nodes, eval adjusted by
//     max(good_move_count, 3) before returning to parent
//   - Iterative deepening with no depth cap when timed

#include "board.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace onitama {

// ------------------------------------------------------------------ //
//  Feature toggles (match Python behaviour when all false)             //
// ------------------------------------------------------------------ //
inline bool AB_USE_PV_MOVES               = false;
inline bool AB_USE_KILLER_MOVES           = false;
inline bool AB_USE_HISTORY_HEURISTIC      = false;
inline bool AB_USE_CENTER_PRIORITY_ORDER  = true;

class ABSearchBot {
public:
    using TimePoint = std::chrono::steady_clock::time_point;

    int max_depth;
    int expansions = 0;

    explicit ABSearchBot(int depth = 5) : max_depth(depth) {}

    // ---- public interface ------------------------------------------------- //

    Move request_move(BoardState board, double time_limit_secs = -1.0) {
        expansions = 0;
        pv_moves_.clear();
        killer_moves_.clear();
        history_.clear();

        // No time limit: single fixed-depth search (Python time_limit=None)
        if (time_limit_secs < 0) {
            auto [eval_val, move] = recursive_search(board, 0, -2000, 2000,
                                                     true, max_depth, nullptr);
            return move.valid() ? move : fallback_move(board);
        }

        // Iterative deepening with time control (matches Python exactly:
        // depth_limit starts at 1 and increments without cap until deadline)
        auto start = std::chrono::steady_clock::now();
        TimePoint deadline = start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(time_limit_secs));
        Move best_move;
        int depth_limit = 1;

        try {
            while (true) {
                auto [eval_val, move] = recursive_search(board, 0, -2000, 2000,
                                                         true, depth_limit, &deadline);
                if (move.valid()) {
                    best_move = move;
                    if (AB_USE_PV_MOVES)
                        pv_moves_[board.turn_num] = move;
                }
                if (std::chrono::steady_clock::now() >= deadline) break;
                depth_limit++;
            }
        } catch (const SearchTimeout&) {
            // time expired mid-search; use last completed best
        }

        return best_move.valid() ? best_move : fallback_move(board);
    }

    /// Like request_move but also returns the search evaluation (red-positive).
    std::pair<int, Move> request_move_with_eval(BoardState board) {
        expansions = 0;
        pv_moves_.clear();
        killer_moves_.clear();
        history_.clear();

        auto [eval_val, move] = recursive_search(board, 0, -2000, 2000,
                                                 true, max_depth, nullptr);
        Move best = move.valid() ? move : fallback_move(board);
        return {eval_val, best};
    }

private:
    struct SearchTimeout {};

    // Move ordering tables (only populated when corresponding toggles are on)
    std::unordered_map<int, Move> pv_moves_;
    std::unordered_map<uint64_t, std::vector<Move>> killer_moves_;
    std::unordered_map<uint64_t, int> history_;

    static uint64_t move_hash(const Move& m) {
        return static_cast<uint64_t>(m.from_sq) |
               (static_cast<uint64_t>(m.to_sq) << 8) |
               (static_cast<uint64_t>(m.card_idx) << 16);
    }

    static uint64_t killer_key(int depth, int turn_num) {
        return (static_cast<uint64_t>(depth) << 32) | static_cast<uint64_t>(turn_num);
    }

    Move fallback_move(const BoardState& board) const {
        auto moves = board.possible_moves_captures_first();
        return moves.empty() ? Move{} : moves[0];
    }

    // ---- move ordering --------------------------------------------------- //

    int move_priority(const Move& m, const BoardState& board, int depth,
                      const Move* pv_move, bool is_capture) const {
        if (AB_USE_PV_MOVES && pv_move && m == *pv_move) return 10000;
        if (is_capture) return 5000;

        if (AB_USE_KILLER_MOVES) {
            auto kk = killer_key(depth, board.turn_num);
            auto kit = killer_moves_.find(kk);
            if (kit != killer_moves_.end()) {
                for (const auto& km : kit->second) {
                    if (m == km) return 1000;
                }
            }
        }

        int x = sq_col(m.to_sq), y = sq_row(m.to_sq);
        int base = AB_USE_CENTER_PRIORITY_ORDER ? BoardState::CENTER_PRIORITY[x][y] : 0;
        int hist = 0;
        if (AB_USE_HISTORY_HEURISTIC) {
            auto hit = history_.find(move_hash(m));
            hist = (hit != history_.end()) ? hit->second : 0;
        }
        return base + hist;
    }

    // Returns a flat move list. When an instant win exists, returns just that
    // one move (matching Python's early-return behaviour). Otherwise returns
    // captures then non-captures, optionally sorted by priority when toggles on.
    std::vector<Move> order_moves(const BoardState::SeparatedMoves& sep,
                                  const BoardState& board, int depth) const {
        // Instant win → single-element list (Python returns [winning_move])
        if (sep.has_instant_win) {
            return { sep.winning_move };
        }

        // When all ordering features are off, return captures then others
        // in their original generation order (matches Python).
        if (!AB_USE_PV_MOVES && !AB_USE_KILLER_MOVES && !AB_USE_HISTORY_HEURISTIC
            && !AB_USE_CENTER_PRIORITY_ORDER) {
            std::vector<Move> out;
            out.reserve(sep.captures.size() + sep.other.size());
            out.insert(out.end(), sep.captures.begin(), sep.captures.end());
            out.insert(out.end(), sep.other.begin(), sep.other.end());
            return out;
        }

        const Move* pv = nullptr;
        if (AB_USE_PV_MOVES) {
            auto pvit = pv_moves_.find(board.turn_num);
            if (pvit != pv_moves_.end()) pv = &pvit->second;
        }

        struct Scored { int score; Move m; };
        std::vector<Scored> all;
        all.reserve(sep.captures.size() + sep.other.size());

        for (const auto& m : sep.captures)
            all.push_back({move_priority(m, board, depth, pv, true), m});
        for (const auto& m : sep.other)
            all.push_back({move_priority(m, board, depth, pv, false), m});

        std::sort(all.begin(), all.end(),
                  [](const Scored& a, const Scored& b) { return a.score > b.score; });

        std::vector<Move> out;
        out.reserve(all.size());
        for (auto& s : all) out.push_back(s.m);
        return out;
    }

    void record_cutoff(const Move& m, const BoardState::SeparatedMoves& sep,
                       const BoardState& board, int depth) {
        if (!AB_USE_KILLER_MOVES && !AB_USE_HISTORY_HEURISTIC) return;

        bool is_capture = false;
        for (const auto& cm : sep.captures) {
            if (m == cm) { is_capture = true; break; }
        }

        if (AB_USE_KILLER_MOVES && !is_capture) {
            auto kk = killer_key(depth, board.turn_num);
            auto& killers = killer_moves_[kk];
            bool found = false;
            for (const auto& km : killers) if (m == km) { found = true; break; }
            if (!found) killers.push_back(m);
        }

        if (AB_USE_HISTORY_HEURISTIC) {
            int boost = (max_depth - depth + 1) * (max_depth - depth + 1);
            history_[move_hash(m)] += boost;
        }
    }

    // ---- core search (matches Python recursive_search exactly) ----------- //
    //
    //  Python signature:
    //    recursive_search(board, depth=0, alpha=-2000, beta=2000,
    //                     return_move=False, max_depth=None, deadline=None)
    //
    //  - depth counts UP from 0, leaf at depth >= max_depth
    //  - possible_moves includes instant wins as single-element lists
    //    which flow through simulate() → evaluate() normally
    //  - Mobility bonus applied at non-root nodes after the loop:
    //    RED:  eval_to_beat += max(good_move_count, 3)
    //    BLUE: eval_to_beat -= max(good_move_count, 3)
    //  - good_move_count conditions:
    //    RED:  alpha > 50  && result >= alpha - 20
    //    BLUE: beta  < -50 && result <= beta  + 20

    std::pair<int, Move> recursive_search(
        BoardState& board, int depth, int alpha, int beta,
        bool return_move, int max_d, const TimePoint* deadline)
    {
        expansions++;
        if (deadline && std::chrono::steady_clock::now() >= *deadline)
            throw SearchTimeout{};

        // Generate moves (instant wins become single-element lists)
        auto sep = board.possible_moves_search_optimised();
        auto possible_moves = order_moves(sep, board, depth);

        Move best_move_so_far;
        bool is_red = board.is_red_turn();

        if (is_red) {
            int eval_to_beat = -2000;
            int good_move_count = 0;

            for (const auto& move : possible_moves) {
                // simulate: apply move, check leaf, recurse/evaluate, undo
                int result = simulate(move, board, depth + 1, alpha, beta,
                                      max_d, deadline);

                if (alpha > 50 && result >= alpha - 20)
                    good_move_count++;

                if (result > eval_to_beat) {
                    eval_to_beat = result;
                    best_move_so_far = move;
                }
                if (eval_to_beat > beta) {
                    record_cutoff(move, sep, board, depth);
                    break;
                }
                alpha = std::max(alpha, eval_to_beat);
            }

            if (!return_move) {
                eval_to_beat += std::max(good_move_count, 3);
                alpha = std::max(alpha, eval_to_beat);
            }

            if (return_move) {
                if (!best_move_so_far.valid() && !possible_moves.empty())
                    best_move_so_far = possible_moves[0];
            } else {
                best_move_so_far = {};
            }

            return {eval_to_beat, best_move_so_far};

        } else {
            int eval_to_beat = 2000;
            int good_move_count = 0;

            for (const auto& move : possible_moves) {
                int result = simulate(move, board, depth + 1, alpha, beta,
                                      max_d, deadline);

                if (beta < -50 && result <= beta + 20)
                    good_move_count++;

                if (result < eval_to_beat) {
                    eval_to_beat = result;
                    best_move_so_far = move;
                }
                if (eval_to_beat < alpha) {
                    record_cutoff(move, sep, board, depth);
                    break;
                }
                beta = std::min(beta, eval_to_beat);
            }

            if (!return_move) {
                eval_to_beat -= std::max(good_move_count, 3);
                beta = std::min(beta, eval_to_beat);
            }

            if (return_move) {
                if (!best_move_so_far.valid() && !possible_moves.empty())
                    best_move_so_far = possible_moves[0];
            } else {
                best_move_so_far = {};
            }

            return {eval_to_beat, best_move_so_far};
        }
    }

    // ---- simulate (matches Python simulate exactly) ---------------------- //
    //
    //  Python:
    //    def simulate(self, move, board, depth, alpha, beta, max_depth, deadline):
    //        undo = board.apply_move(move)
    //        try:
    //            terminal = board.is_won()
    //            if depth >= max_depth or terminal:
    //                return self._evaluate_position(board, depth)
    //            else:
    //                return self.recursive_search(board, depth, alpha, beta,
    //                                             max_depth=max_depth, deadline=deadline)[0]
    //        finally:
    //            board.undo_move(undo)
    //
    //  Note: 'depth' here is already depth+1 from the caller.

    int simulate(const Move& m, BoardState& board, int depth,
                 int alpha, int beta, int max_d, const TimePoint* deadline)
    {
        if (deadline && std::chrono::steady_clock::now() >= *deadline)
            throw SearchTimeout{};

        auto undo = board.apply_move(m);
        int result;

        if (depth >= max_d || board.is_terminal()) {
            result = board.evaluate(depth);
        } else {
            result = recursive_search(board, depth, alpha, beta,
                                      false, max_d, deadline).first;
        }

        board.undo_move(undo);
        return result;
    }
};

}  // namespace onitama
