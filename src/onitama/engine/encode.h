#pragma once
// encode.h — Board → NN tensor encoding
//
// Produces the same encoding as encode_state_for_value_net() in VN1_train.py:
//
//   board_tensor: float[4][5][5]
//     ch 0: friendly master
//     ch 1: friendly students
//     ch 2: enemy master
//     ch 3: enemy students
//   All from the moving player's perspective (blue coords flipped via (4-x, 4-y))
//
//   cards_tensor: float[3][16]
//     row 0: player's 2 cards  (two 1s)
//     row 1: opponent's 2 cards
//     row 2: transition card    (one 1)

#include "board.h"

#include <array>
#include <cstring>

namespace onitama {

struct BoardEncoding {
    float board[4][5][5];  // [channel][row][col]
    float cards[3][16];
};

inline BoardEncoding encode_for_nn(const BoardState& b) {
    BoardEncoding enc;
    std::memset(&enc, 0, sizeof(enc));

    bool blue_to_move = !b.is_red_turn();

    uint32_t f_master, f_students, e_master, e_students;
    int p_card0, p_card1, o_card0, o_card1;

    if (blue_to_move) {
        f_master   = b.blue_master;
        f_students = b.blue_students;
        e_master   = b.red_master;
        e_students = b.red_students;
        p_card0 = b.cards[2]; p_card1 = b.cards[3];
        o_card0 = b.cards[0]; o_card1 = b.cards[1];
    } else {
        f_master   = b.red_master;
        f_students = b.red_students;
        e_master   = b.blue_master;
        e_students = b.blue_students;
        p_card0 = b.cards[0]; p_card1 = b.cards[1];
        o_card0 = b.cards[2]; o_card1 = b.cards[3];
    }
    int trans = b.cards[4];

    auto place = [&](int channel, uint32_t bits) {
        while (bits) {
            int s = BoardState::lsb(bits);
            bits &= bits - 1;
            int x = sq_col(s), y = sq_row(s);
            if (blue_to_move) {
                x = 4 - x;
                y = 4 - y;
            }
            enc.board[channel][y][x] = 1.0f;
        }
    };

    place(0, f_master);
    place(1, f_students);
    place(2, e_master);
    place(3, e_students);

    enc.cards[0][p_card0] = 1.0f;
    enc.cards[0][p_card1] = 1.0f;
    enc.cards[1][o_card0] = 1.0f;
    enc.cards[1][o_card1] = 1.0f;
    enc.cards[2][trans]   = 1.0f;

    return enc;
}

}  // namespace onitama
