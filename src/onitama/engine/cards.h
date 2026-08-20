#pragma once
// cards.h — Static card definitions and precomputed attack tables for Onitama
//
// Bit index convention:  sq = row * 5 + col   (row 0 = bottom for Red)
//   col 0..4  →  x 0..4
//   row 0..4  →  y 0..4
//
// Card offsets are stored in Red-perspective (dx, dy).
// Red destination:  (x - dx, y + dy)   — note the negated dx!
// Blue destination: (x + dx, y - dy)

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace onitama {

// ------------------------------------------------------------------ //
//  Card definitions                                                   //
// ------------------------------------------------------------------ //

struct Offset {
    int dx, dy;
};

struct CardDef {
    std::string_view name;
    int idx;
    int num_offsets;
    std::array<Offset, 4> offsets;  // max 4 offsets per card
};

// Matches Deck.py exactly (indices 0–15)
inline constexpr std::array<CardDef, 16> DECK = {{
    {"tiger",    0, 2, {{{ 0, 2},{ 0,-1},{ 0, 0},{ 0, 0}}}},
    {"monkey",   1, 4, {{{ 1, 1},{-1, 1},{-1,-1},{ 1,-1}}}},
    {"dragon",   2, 4, {{{ 2, 1},{-2, 1},{-1,-1},{ 1,-1}}}},
    {"crab",     3, 3, {{{ 0, 1},{ 2, 0},{-2, 0},{ 0, 0}}}},
    {"mantis",   4, 3, {{{ 1, 1},{-1, 1},{ 0,-1},{ 0, 0}}}},
    {"frog",     5, 3, {{{-1, 1},{-2, 0},{ 1,-1},{ 0, 0}}}},
    {"elephant", 6, 4, {{{-1, 1},{ 1, 1},{-1, 0},{ 1, 0}}}},
    {"rooster",  7, 4, {{{ 1, 1},{-1, 0},{ 1, 0},{-1,-1}}}},
    {"boar",     8, 3, {{{ 0, 1},{-1, 0},{ 1, 0},{ 0, 0}}}},
    {"ox",       9, 3, {{{ 0, 1},{ 1, 0},{ 0,-1},{ 0, 0}}}},
    {"crane",   10, 3, {{{ 0, 1},{ 1,-1},{-1,-1},{ 0, 0}}}},
    {"eel",     11, 3, {{{-1, 1},{ 1, 0},{-1,-1},{ 0, 0}}}},
    {"horse",   12, 3, {{{ 0, 1},{-1, 0},{ 0,-1},{ 0, 0}}}},
    {"cobra",   13, 3, {{{ 1, 1},{-1, 0},{ 1,-1},{ 0, 0}}}},
    {"goose",   14, 4, {{{-1, 1},{-1, 0},{ 1, 0},{ 1,-1}}}},
    {"rabbit",  15, 3, {{{ 1, 1},{ 2, 0},{-1,-1},{ 0, 0}}}},
}};

// ------------------------------------------------------------------ //
//  Precomputed attack tables                                          //
// ------------------------------------------------------------------ //
//  ATTACKS[card_idx][colour][from_sq]  →  bitmask of reachable squares
//  colour: 0 = RED, 1 = BLUE

inline constexpr int sq(int col, int row) { return row * 5 + col; }
inline constexpr int sq_col(int s) { return s % 5; }
inline constexpr int sq_row(int s) { return s / 5; }

struct AttackTables {
    // [16 cards][2 colours][25 from-squares]
    uint32_t table[16][2][25];

    constexpr AttackTables() : table{} {
        for (int c = 0; c < 16; ++c) {
            const auto& card = DECK[c];
            for (int from = 0; from < 25; ++from) {
                int fx = sq_col(from);
                int fy = sq_row(from);

                uint32_t red_mask = 0;
                uint32_t blue_mask = 0;

                for (int i = 0; i < card.num_offsets; ++i) {
                    int dx = card.offsets[i].dx;
                    int dy = card.offsets[i].dy;

                    // Red: dest = (x - dx, y + dy)
                    int rx = fx - dx;
                    int ry = fy + dy;
                    if (rx >= 0 && rx <= 4 && ry >= 0 && ry <= 4) {
                        red_mask |= (1u << sq(rx, ry));
                    }

                    // Blue: dest = (x + dx, y - dy)
                    int bx = fx + dx;
                    int by = fy - dy;
                    if (bx >= 0 && bx <= 4 && by >= 0 && by <= 4) {
                        blue_mask |= (1u << sq(bx, by));
                    }
                }

                table[c][0][from] = red_mask;
                table[c][1][from] = blue_mask;
            }
        }
    }
};

inline constexpr AttackTables ATTACKS{};

}  // namespace onitama
