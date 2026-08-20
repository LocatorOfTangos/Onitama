#pragma once
// board.h — Bitboard-based BoardState for Onitama
//
// Piece layout (matching Python Boardstate.positions):
//   Index 0   = red master
//   Index 1-4 = red students
//   Index 5   = blue master
//   Index 6-9 = blue students
//
// Bitboard: uint32_t  with  bit index = row * 5 + col   (bits 0..24 used)
//   red_master   : 1 bit set (or 0 if captured)
//   red_students : 0-4 bits set
//   blue_master  : 1 bit set (or 0 if captured)
//   blue_students: 0-4 bits set

#include "cards.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <utility>
#include <vector>

namespace onitama {

// ------------------------------------------------------------------ //
//  Move                                                               //
// ------------------------------------------------------------------ //

struct Move {
    uint8_t from_sq;   // 0-24
    uint8_t to_sq;     // 0-24
    uint8_t card_idx;  // 0-15

    Move() : from_sq(255), to_sq(255), card_idx(255) {}
    Move(int f, int t, int c)
        : from_sq(static_cast<uint8_t>(f)),
          to_sq(static_cast<uint8_t>(t)),
          card_idx(static_cast<uint8_t>(c)) {}

    bool operator==(const Move& o) const {
        return from_sq == o.from_sq && to_sq == o.to_sq && card_idx == o.card_idx;
    }
    bool operator!=(const Move& o) const { return !(*this == o); }
    bool valid() const { return from_sq != 255; }

    // Convert to Python-compatible ((fx, fy), (tx, ty), card_idx)
    auto to_coords() const {
        struct Coords {
            int fx, fy, tx, ty, card;
        };
        return Coords{sq_col(from_sq), sq_row(from_sq),
                      sq_col(to_sq), sq_row(to_sq),
                      card_idx};
    }
};

struct MoveHash {
    std::size_t operator()(const Move& m) const {
        return static_cast<std::size_t>(m.from_sq) |
               (static_cast<std::size_t>(m.to_sq) << 8) |
               (static_cast<std::size_t>(m.card_idx) << 16);
    }
};

// ------------------------------------------------------------------ //
//  Win result                                                         //
// ------------------------------------------------------------------ //

enum class WinColour : uint8_t { NONE = 0, RED = 1, BLUE = 2 };
enum class WinMethod : uint8_t { NONE = 0, STONE = 1, STREAM = 2 };

struct WinResult {
    WinColour colour = WinColour::NONE;
    WinMethod method = WinMethod::NONE;
    explicit operator bool() const { return colour != WinColour::NONE; }
};

// ------------------------------------------------------------------ //
//  UndoRecord                                                         //
// ------------------------------------------------------------------ //

struct UndoRecord {
    uint32_t prev_red_master;
    uint32_t prev_red_students;
    uint32_t prev_blue_master;
    uint32_t prev_blue_students;
    std::array<uint8_t, 5> prev_cards;  // [red0, red1, blue0, blue1, trans]
    int prev_turn_num;
};

// ------------------------------------------------------------------ //
//  BoardState                                                         //
// ------------------------------------------------------------------ //

class BoardState {
public:
    uint32_t red_master    = 0;
    uint32_t red_students  = 0;
    uint32_t blue_master   = 0;
    uint32_t blue_students = 0;

    // cards[0..1] = red's cards, cards[2..3] = blue's cards, cards[4] = transition
    std::array<uint8_t, 5> cards{};

    int  turn_num  = 0;
    bool red_start = true;

    // ---- constructors ---- //

    BoardState() = default;

    // Construct from explicit state
    BoardState(uint32_t rm, uint32_t rs, uint32_t bm, uint32_t bs,
               std::array<uint8_t, 5> c, int tn, bool rs_flag)
        : red_master(rm), red_students(rs), blue_master(bm), blue_students(bs),
          cards(c), turn_num(tn), red_start(rs_flag) {}

    // Construct a random starting position (random cards, random starting player)
    static BoardState random_start(std::mt19937& rng) {
        BoardState b;
        // Red: master at (2,0), students at (0,0),(1,0),(3,0),(4,0)
        b.red_master   = 1u << sq(2, 0);
        b.red_students = (1u << sq(0, 0)) | (1u << sq(1, 0)) |
                         (1u << sq(3, 0)) | (1u << sq(4, 0));
        // Blue: master at (2,4), students at (0,4),(1,4),(3,4),(4,4)
        b.blue_master   = 1u << sq(2, 4);
        b.blue_students = (1u << sq(0, 4)) | (1u << sq(1, 4)) |
                          (1u << sq(3, 4)) | (1u << sq(4, 4));

        // Deal 5 random cards from the 16-card deck
        std::array<uint8_t, 16> indices;
        for (int i = 0; i < 16; ++i) indices[i] = static_cast<uint8_t>(i);
        std::shuffle(indices.begin(), indices.end(), rng);
        b.cards = {indices[0], indices[1], indices[2], indices[3], indices[4]};

        // Random starting player
        std::uniform_int_distribution<int> coin(0, 1);
        b.red_start = (coin(rng) == 0);

        return b;
    }

    // ---- queries ---- //

    bool is_red_turn() const {
        bool turn_odd = (turn_num % 2 != 0);
        return red_start ^ turn_odd;
    }

    // Occupancy helpers
    uint32_t red_all()  const { return red_master | red_students; }
    uint32_t blue_all() const { return blue_master | blue_students; }
    uint32_t all_occ()  const { return red_all() | blue_all(); }

    uint32_t friendly_occ() const { return is_red_turn() ? red_all()  : blue_all(); }
    uint32_t enemy_occ()    const { return is_red_turn() ? blue_all() : red_all(); }

    WinResult is_won() const {
        // Red stream: red master on (2,4) = bit 22
        if (red_master & (1u << sq(2, 4)))
            return {WinColour::RED, WinMethod::STREAM};
        // Blue stream: blue master on (2,0) = bit 2
        if (blue_master & (1u << sq(2, 0)))
            return {WinColour::BLUE, WinMethod::STREAM};
        // Stone: master captured (bitboard == 0)
        if (red_master == 0)
            return {WinColour::BLUE, WinMethod::STONE};
        if (blue_master == 0)
            return {WinColour::RED, WinMethod::STONE};
        return {};
    }

    bool is_terminal() const { return is_won().colour != WinColour::NONE; }

    // ---- move generation ---- //

    // Iterate set bits
    static inline int lsb(uint32_t v) { return __builtin_ctz(v); }
    static inline uint32_t pop_lsb(uint32_t& v) {
        uint32_t b = v & (-v);
        v ^= b;
        return b;
    }

    // Generate all legal moves (unordered)
    std::vector<Move> possible_moves() const {
        bool red = is_red_turn();
        int colour = red ? 0 : 1;
        uint32_t own = friendly_occ();
        uint32_t pieces = own;

        int c0 = red ? cards[0] : cards[2];
        int c1 = red ? cards[1] : cards[3];

        std::vector<Move> moves;
        moves.reserve(40);

        while (pieces) {
            int from = lsb(pieces);
            pieces &= pieces - 1;  // clear lsb

            uint32_t dests0 = ATTACKS.table[c0][colour][from] & ~own;
            uint32_t dests1 = ATTACKS.table[c1][colour][from] & ~own;

            uint32_t d = dests0;
            while (d) {
                int to = lsb(d);
                d &= d - 1;
                moves.emplace_back(from, to, c0);
            }
            d = dests1;
            while (d) {
                int to = lsb(d);
                d &= d - 1;
                moves.emplace_back(from, to, c1);
            }
        }
        return moves;
    }

    // Search-optimised: returns immediately on winning move.
    // Separates captures from non-captures.
    struct SeparatedMoves {
        std::vector<Move> captures;
        std::vector<Move> other;
        bool has_instant_win = false;
        Move winning_move;
    };

    SeparatedMoves possible_moves_search_optimised() const {
        bool red = is_red_turn();
        int colour = red ? 0 : 1;
        uint32_t own  = friendly_occ();
        uint32_t enemy = enemy_occ();
        uint32_t enemy_master = red ? blue_master : red_master;
        uint32_t own_master   = red ? red_master  : blue_master;
        uint32_t stream_sq    = red ? (1u << sq(2, 4)) : (1u << sq(2, 0));

        int c0 = red ? cards[0] : cards[2];
        int c1 = red ? cards[1] : cards[3];

        SeparatedMoves result;
        result.captures.reserve(20);
        result.other.reserve(20);

        auto gen_for_piece = [&](int from, bool is_master) {
            for (int card : {c0, c1}) {
                uint32_t dests = ATTACKS.table[card][colour][from] & ~own;

                while (dests) {
                    int to = lsb(dests);
                    dests &= dests - 1;
                    uint32_t to_bit = 1u << to;

                    // Instant win: capture enemy master
                    if (to_bit & enemy_master) {
                        result.has_instant_win = true;
                        result.winning_move = Move(from, to, card);
                        return;
                    }
                    // Instant win: own master reaches stream
                    if (is_master && (to_bit & stream_sq)) {
                        result.has_instant_win = true;
                        result.winning_move = Move(from, to, card);
                        return;
                    }

                    if (to_bit & enemy) {
                        result.captures.emplace_back(from, to, card);
                    } else {
                        result.other.emplace_back(from, to, card);
                    }
                }
                if (result.has_instant_win) return;
            }
        };

        // Process master first
        if (own_master) {
            gen_for_piece(lsb(own_master), true);
            if (result.has_instant_win) return result;
        }
        // Then students
        uint32_t students = red ? red_students : blue_students;
        while (students) {
            int from = lsb(students);
            students &= students - 1;
            gen_for_piece(from, false);
            if (result.has_instant_win) return result;
        }

        return result;
    }

    // Flat list with captures first (convenience for basic AB)
    std::vector<Move> possible_moves_captures_first() const {
        auto sep = possible_moves_search_optimised();
        if (sep.has_instant_win) return {sep.winning_move};
        std::vector<Move> out;
        out.reserve(sep.captures.size() + sep.other.size());
        out.insert(out.end(), sep.captures.begin(), sep.captures.end());
        out.insert(out.end(), sep.other.begin(), sep.other.end());
        return out;
    }

    // ---- apply / undo ---- //

    UndoRecord apply_move(const Move& m) {
        UndoRecord u;
        u.prev_red_master   = red_master;
        u.prev_red_students = red_students;
        u.prev_blue_master  = blue_master;
        u.prev_blue_students = blue_students;
        u.prev_cards        = cards;
        u.prev_turn_num     = turn_num;

        bool red = is_red_turn();
        uint32_t from_bit = 1u << m.from_sq;
        uint32_t to_bit   = 1u << m.to_sq;

        // Move the piece
        if (red) {
            if (red_master & from_bit) {
                red_master = to_bit;
            } else {
                red_students ^= from_bit;
                red_students |= to_bit;
            }
            // Capture enemy piece at destination
            if (blue_master & to_bit)   blue_master = 0;
            if (blue_students & to_bit) blue_students ^= to_bit;

            // Swap card: find which of red's cards was used
            if (cards[0] == m.card_idx) {
                std::swap(cards[0], cards[4]);
            } else {
                std::swap(cards[1], cards[4]);
            }
        } else {
            if (blue_master & from_bit) {
                blue_master = to_bit;
            } else {
                blue_students ^= from_bit;
                blue_students |= to_bit;
            }
            // Capture enemy piece at destination
            if (red_master & to_bit)   red_master = 0;
            if (red_students & to_bit) red_students ^= to_bit;

            // Swap card
            if (cards[2] == m.card_idx) {
                std::swap(cards[2], cards[4]);
            } else {
                std::swap(cards[3], cards[4]);
            }
        }

        turn_num++;
        return u;
    }

    void undo_move(const UndoRecord& u) {
        red_master   = u.prev_red_master;
        red_students = u.prev_red_students;
        blue_master  = u.prev_blue_master;
        blue_students = u.prev_blue_students;
        cards        = u.prev_cards;
        turn_num     = u.prev_turn_num;
    }

    // ---- heuristic evaluation (RED-positive) ---- //

    static constexpr int CENTER_PRIORITY[5][5] = {
        {-10,  0, 10,  0, -10},
        {  0, 10, 20, 10,   0},
        { 10, 20, 30, 20,  10},
        {  0, 10, 20, 10,   0},
        {-10,  0, 10,  0, -10},
    };

    static constexpr int OPENING_MASTER_POS[5][5] = {
        {  0, 0,-20,-60,-100},
        {  0, 0,-20,-60,-100},
        {  0, 0,-20,-60,-100},
        {  0, 0,-20,-60,-100},
        {  0, 0,-20,-60,-100},
    };

    static constexpr int MIDGAME_MASTER_POS[5][5] = {
        {-20,  0,  0,-40,-80},
        {-20,  0,  0,-20,-40},
        {-20,  0,  0,-20,0},
        {-20,  0,  0,-20,-40},
        {-20,  0,  0,-40,-80},
    };

    static constexpr int ENDGAME_MASTER_POS[5][5] = {
        {-120,-80, 20, 60, 60},
        {-100,-60, 40, 120,120},
        {-100,-60, 60,120,0},
        {-100,-60, 40, 120,120},
        {-120,-80, 20, 60, 60},
    };

    static constexpr int STUDENT_VALUE = 50;

    int evaluate(int depth = 0) const {
        auto w = is_won();
        if (w) {
            int dd = depth * 5;
            return (w.colour == WinColour::RED) ? (1000 - dd) : (-1000 + dd);
        }

        int score = 0;

        // Red students
        uint32_t rs = red_students;
        while (rs) {
            int s = lsb(rs); rs &= rs - 1;
            int x = sq_col(s), y = sq_row(s);
            score += CENTER_PRIORITY[x][y] + STUDENT_VALUE;
        }
        // Blue students
        uint32_t bs = blue_students;
        while (bs) {
            int s = lsb(bs); bs &= bs - 1;
            int x = sq_col(s), y = sq_row(s);
            score -= CENTER_PRIORITY[x][y] + STUDENT_VALUE;
        }

        // Game stage
        int n_red  = __builtin_popcount(red_students);
        int n_blue = __builtin_popcount(blue_students);
        int min_s  = std::min(n_red, n_blue);
        // 0 = OPENING, 1 = MIDGAME, 2 = ENDGAME
        int stage = (min_s >= 4) ? 0 : (min_s >= 2) ? 1 : 2;

        // Red master position value
        if (red_master) {
            int s = lsb(red_master);
            int rx = sq_col(s), ry = sq_row(s);
            if      (stage == 0) score += OPENING_MASTER_POS[rx][ry];
            else if (stage == 1) score += MIDGAME_MASTER_POS[rx][ry];
            else                 score += ENDGAME_MASTER_POS[rx][ry];
        }
        // Blue master position value (mirrored: [4-bx][4-by])
        if (blue_master) {
            int s = lsb(blue_master);
            int bx = sq_col(s), by = sq_row(s);
            if      (stage == 0) score -= OPENING_MASTER_POS[4 - bx][4 - by];
            else if (stage == 1) score -= MIDGAME_MASTER_POS[4 - bx][4 - by];
            else                 score -= ENDGAME_MASTER_POS[4 - bx][4 - by];
        }

        return score;
    }

    // ---- student count helpers ---- //

    int red_student_count()  const { return __builtin_popcount(red_students); }
    int blue_student_count() const { return __builtin_popcount(blue_students); }
};

}  // namespace onitama
