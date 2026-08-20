// onitama_engine.cpp — pybind11 bindings for the C++ Onitama engine
//
// Exposes: BoardState, Move, MinimaxTree, ABSearchBot, MCTSBot
// Plus: from_python_board() factory to convert Python Boardstate → C++ BoardState

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "board.h"
#include "cards.h"
#include "encode.h"
#include "minimax.h"
#include "absearch.h"
#include "mcts.h"
#include "value_mcts.h"

#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace onitama;

// ------------------------------------------------------------------ //
//  Helper: Convert Python Boardstate → C++ BoardState                 //
// ------------------------------------------------------------------ //

// Python boardstate has:
//   .positions: list of 10 tuples (x,y) or None — [rm, rs1..rs4, bm, bs1..bs4]
//   .red_cards: tuple of 2 Card objects with .name
//   .blue_cards: tuple of 2 Card objects with .name
//   .trans_card: Card object with .name
//   .turn_num: int
//   .red_start: bool

static int card_name_to_index(const std::string& name) {
    for (int i = 0; i < 16; ++i) {
        if (DECK[i].name == name) return i;
    }
    throw std::runtime_error("Unknown card name: " + name);
}

static BoardState from_python_board(py::object py_board) {
    BoardState b;

    // Positions
    py::list positions = py_board.attr("positions");
    auto set_master = [](uint32_t& bb, py::object pos) {
        if (pos.is_none()) {
            bb = 0;
        } else {
            py::tuple t = pos.cast<py::tuple>();
            int x = t[0].cast<int>();
            int y = t[1].cast<int>();
            bb = 1u << sq(x, y);
        }
    };
    auto add_student = [](uint32_t& bb, py::object pos) {
        if (!pos.is_none()) {
            py::tuple t = pos.cast<py::tuple>();
            int x = t[0].cast<int>();
            int y = t[1].cast<int>();
            bb |= 1u << sq(x, y);
        }
    };

    b.red_master = 0;
    b.red_students = 0;
    b.blue_master = 0;
    b.blue_students = 0;

    set_master(b.red_master, positions[0]);
    for (int i = 1; i <= 4; ++i) add_student(b.red_students, positions[i]);
    set_master(b.blue_master, positions[5]);
    for (int i = 6; i <= 9; ++i) add_student(b.blue_students, positions[i]);

    // Cards
    py::tuple rc = py_board.attr("red_cards");
    py::tuple bc = py_board.attr("blue_cards");
    py::object tc = py_board.attr("trans_card");

    b.cards[0] = static_cast<uint8_t>(card_name_to_index(rc[0].attr("name").cast<std::string>()));
    b.cards[1] = static_cast<uint8_t>(card_name_to_index(rc[1].attr("name").cast<std::string>()));
    b.cards[2] = static_cast<uint8_t>(card_name_to_index(bc[0].attr("name").cast<std::string>()));
    b.cards[3] = static_cast<uint8_t>(card_name_to_index(bc[1].attr("name").cast<std::string>()));
    b.cards[4] = static_cast<uint8_t>(card_name_to_index(tc.attr("name").cast<std::string>()));

    b.turn_num = py_board.attr("turn_num").cast<int>();
    b.red_start = py_board.attr("red_start").cast<bool>();

    return b;
}

// ------------------------------------------------------------------ //
//  Helper: Convert C++ BoardState → Python Boardstate                 //
// ------------------------------------------------------------------ //

static py::object to_python_board(const BoardState& b, py::module_& card_module) {
    // Build positions list
    py::list positions;

    auto bit_to_coord = [](uint32_t bb) -> py::object {
        if (bb == 0) return py::none();
        int s = __builtin_ctz(bb);
        return py::make_tuple(sq_col(s), sq_row(s));
    };

    // Red master
    positions.append(bit_to_coord(b.red_master));
    // Red students (up to 4)
    uint32_t rs = b.red_students;
    int rs_count = 0;
    while (rs) {
        int s = BoardState::lsb(rs);
        rs &= rs - 1;
        positions.append(py::make_tuple(sq_col(s), sq_row(s)));
        rs_count++;
    }
    for (int i = rs_count; i < 4; ++i) positions.append(py::none());

    // Blue master
    positions.append(bit_to_coord(b.blue_master));
    // Blue students (up to 4)
    uint32_t bst = b.blue_students;
    int bs_count = 0;
    while (bst) {
        int s = BoardState::lsb(bst);
        bst &= bst - 1;
        positions.append(py::make_tuple(sq_col(s), sq_row(s)));
        bs_count++;
    }
    for (int i = bs_count; i < 4; ++i) positions.append(py::none());

    // Cards: construct Card(name, idx, moveset) from C++ deck metadata
    auto get_card = [&card_module](uint8_t idx) -> py::object {
        const auto& def = DECK[idx];
        py::tuple moveset(def.num_offsets);
        for (int i = 0; i < def.num_offsets; ++i) {
            moveset[i] = py::make_tuple(def.offsets[i].dx, def.offsets[i].dy);
        }
        return card_module.attr("Card")(py::str(def.name), def.idx, moveset);
    };

    py::tuple red_cards = py::make_tuple(get_card(b.cards[0]), get_card(b.cards[1]));
    py::tuple blue_cards = py::make_tuple(get_card(b.cards[2]), get_card(b.cards[3]));
    py::object trans_card = get_card(b.cards[4]);

    // Import Boardstate class
    py::module_ bs_module = py::module_::import("onitama.Boardstate");
    py::object Boardstate = bs_module.attr("Boardstate");

    return Boardstate(
        py::arg("positions") = positions,
        py::arg("red_cards") = red_cards,
        py::arg("blue_cards") = blue_cards,
        py::arg("trans_card") = trans_card,
        py::arg("turn_num") = b.turn_num,
        py::arg("red_start") = b.red_start
    );
}

// ------------------------------------------------------------------ //
//  Module definition                                                  //
// ------------------------------------------------------------------ //

PYBIND11_MODULE(onitama_engine, m) {
    m.doc() = "C++ Onitama engine with bitboard representation";

    // -- Move --
    py::class_<Move>(m, "Move")
        .def(py::init<>())
        .def(py::init<int, int, int>(), py::arg("from_sq"), py::arg("to_sq"), py::arg("card_idx"))
        .def_readwrite("from_sq", &Move::from_sq)
        .def_readwrite("to_sq", &Move::to_sq)
        .def_readwrite("card_idx", &Move::card_idx)
        .def("valid", &Move::valid)
        .def("to_coords", [](const Move& m) {
            auto c = m.to_coords();
            return py::make_tuple(
                py::make_tuple(c.fx, c.fy),
                py::make_tuple(c.tx, c.ty),
                c.card
            );
        })
        .def("__repr__", [](const Move& m) {
            std::ostringstream os;
            os << "Move(" << (int)m.from_sq << " -> " << (int)m.to_sq
               << ", card=" << (int)m.card_idx << ")";
            return os.str();
        })
        .def("__eq__", &Move::operator==)
        .def("__hash__", [](const Move& m) {
            return MoveHash{}(m);
        });

    // -- WinColour --
    py::enum_<WinColour>(m, "WinColour")
        .value("NONE", WinColour::NONE)
        .value("RED", WinColour::RED)
        .value("BLUE", WinColour::BLUE);

    // -- WinMethod --
    py::enum_<WinMethod>(m, "WinMethod")
        .value("NONE", WinMethod::NONE)
        .value("STONE", WinMethod::STONE)
        .value("STREAM", WinMethod::STREAM);

    // -- WinResult --
    py::class_<WinResult>(m, "WinResult")
        .def_readwrite("colour", &WinResult::colour)
        .def_readwrite("method", &WinResult::method)
        .def("__bool__", [](const WinResult& w) { return bool(w); });

    // -- BoardState --
    py::class_<BoardState>(m, "BoardState")
        .def(py::init<>())
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t,
                       std::array<uint8_t, 5>, int, bool>(),
             py::arg("red_master"), py::arg("red_students"),
             py::arg("blue_master"), py::arg("blue_students"),
             py::arg("cards"), py::arg("turn_num"), py::arg("red_start"))
        .def_readwrite("red_master", &BoardState::red_master)
        .def_readwrite("red_students", &BoardState::red_students)
        .def_readwrite("blue_master", &BoardState::blue_master)
        .def_readwrite("blue_students", &BoardState::blue_students)
        .def_readwrite("cards", &BoardState::cards)
        .def_readwrite("turn_num", &BoardState::turn_num)
        .def_readwrite("red_start", &BoardState::red_start)
        .def("copy", [](const BoardState& b) { return BoardState(b); },
             "Return a copy of this BoardState")
        .def("is_red_turn", &BoardState::is_red_turn)
        .def("is_won", &BoardState::is_won)
        .def("is_terminal", &BoardState::is_terminal)
        .def("evaluate", &BoardState::evaluate, py::arg("depth") = 0)
        .def("possible_moves", &BoardState::possible_moves)
        .def("possible_moves_captures_first", &BoardState::possible_moves_captures_first)
        .def("apply_move", &BoardState::apply_move, py::arg("move"))
        .def("undo_move", &BoardState::undo_move, py::arg("undo_record"))
        .def("red_student_count", &BoardState::red_student_count)
        .def("blue_student_count", &BoardState::blue_student_count)
        .def_static("random_start", [](unsigned seed) {
            std::mt19937 rng(seed);
            return BoardState::random_start(rng);
        }, py::arg("seed"))
        .def_static("random_start_auto", []() {
            std::mt19937 rng(std::random_device{}());
            return BoardState::random_start(rng);
        })
        .def_static("from_python", &from_python_board,
                     py::arg("py_board"),
                     "Convert a Python Boardstate to a C++ BoardState")
        .def("to_python", [](const BoardState& b) {
            py::module_ card_module = py::module_::import("onitama.Card");
            return to_python_board(b, card_module);
        }, "Convert this C++ BoardState to a Python Boardstate")
        .def("encode_for_nn", [](const BoardState& b) {
            auto enc = encode_for_nn(b);
            // Return as dict of numpy arrays
            py::array_t<float> board_arr({4, 5, 5});
            py::array_t<float> cards_arr({3, 16});
            std::memcpy(board_arr.mutable_data(), &enc.board, 100 * sizeof(float));
            std::memcpy(cards_arr.mutable_data(), &enc.cards, 48 * sizeof(float));
            return py::make_tuple(board_arr, cards_arr);
        })
        .def("__repr__", [](const BoardState& b) {
            std::ostringstream os;
            os << "BoardState(turn=" << b.turn_num
               << ", red_turn=" << b.is_red_turn()
               << ", red_pieces=" << __builtin_popcount(b.red_master | b.red_students)
               << ", blue_pieces=" << __builtin_popcount(b.blue_master | b.blue_students)
               << ")";
            return os.str();
        });

    // -- UndoRecord --
    py::class_<UndoRecord>(m, "UndoRecord")
        .def(py::init<>());

    // -- MinimaxTree --
    py::class_<MinimaxTree>(m, "MinimaxTree")
        .def_static("build", &MinimaxTree::build,
                     py::arg("board"), py::arg("max_depth"))
        .def("get_leaf_tensors", [](const MinimaxTree& tree) {
            auto lt = tree.get_leaf_tensors();
            py::array_t<float> boards({lt.count, 4, 5, 5});
            py::array_t<float> cards({lt.count, 3, 16});
            if (lt.count > 0) {
                std::memcpy(boards.mutable_data(), lt.boards.data(),
                            lt.boards.size() * sizeof(float));
                std::memcpy(cards.mutable_data(), lt.cards.data(),
                            lt.cards.size() * sizeof(float));
            }
            return py::make_tuple(boards, cards, lt.count);
        })
        .def("set_leaf_values", [](MinimaxTree& tree, py::array_t<float> values) {
            py::buffer_info buf = values.request();
            tree.set_leaf_values(static_cast<const float*>(buf.ptr),
                                 static_cast<int>(buf.size));
        })
        .def("backup", &MinimaxTree::backup)
        .def("best_move", &MinimaxTree::best_move)
        .def("select_move_temperature", [](const MinimaxTree& tree, double temperature) {
            std::mt19937 rng(std::random_device{}());
            return tree.select_move_temperature(temperature, rng);
        }, py::arg("temperature"))
        .def("get_training_examples", [](const MinimaxTree& tree, int example_depth) {
            auto examples = tree.get_training_examples(example_depth);
            int n = static_cast<int>(examples.size());
            py::array_t<float> boards({n, 4, 5, 5});
            py::array_t<float> cards({n, 3, 16});
            py::array_t<float> values(n);

            auto* b_ptr = boards.mutable_data();
            auto* c_ptr = cards.mutable_data();
            auto* v_ptr = values.mutable_data();

            for (int i = 0; i < n; ++i) {
                std::memcpy(b_ptr + i * 100, &examples[i].encoding.board,
                            100 * sizeof(float));
                std::memcpy(c_ptr + i * 48, &examples[i].encoding.cards,
                            48 * sizeof(float));
                v_ptr[i] = examples[i].value;
            }

            return py::make_tuple(boards, cards, values);
        }, py::arg("example_depth"))
        .def_property_readonly("node_count", [](const MinimaxTree& t) {
            return static_cast<int>(t.nodes.size());
        })
        .def_property_readonly("leaf_count", [](const MinimaxTree& t) {
            return static_cast<int>(t.leaf_indices.size());
        });

    // -- ValueLeafMCTSTree --
    py::class_<ValueLeafMCTSTree>(m, "ValueLeafMCTSTree")
        .def(py::init<const BoardState&, double, float>(),
             py::arg("board"), py::arg("exploration") = 1.4,
             py::arg("virtual_loss") = 0.0f)
        // select_and_expand() → tuple(node_idx, is_terminal, terminal_value, board_np[4,5,5], cards_np[3,16])
        // Board/cards arrays are always returned (zeroed for terminal nodes) to keep the
        // hot-path free of None checks on the Python side.
        .def("select_and_expand", [](ValueLeafMCTSTree& tree) {
            auto res = tree.select_and_expand();
            py::array_t<float> board_arr({4, 5, 5});
            py::array_t<float> cards_arr({3, 16});
            std::memcpy(board_arr.mutable_data(), &res.encoding.board, 100 * sizeof(float));
            std::memcpy(cards_arr.mutable_data(), &res.encoding.cards, 48 * sizeof(float));
            return py::make_tuple(
                res.node_idx,
                res.is_terminal,
                res.terminal_value,
                board_arr,
                cards_arr
            );
        })
        .def("backpropagate", &ValueLeafMCTSTree::backpropagate,
             py::arg("node_idx"), py::arg("value"))
        .def("root_value", &ValueLeafMCTSTree::root_value)
        .def("root_child_visits", [](const ValueLeafMCTSTree& tree) {
            auto cv = tree.root_child_visits();
            py::list out;
            for (auto& [move, visits] : cv) {
                out.append(py::make_tuple(move, visits));
            }
            return out;
        })
        .def("best_move_greedy", &ValueLeafMCTSTree::best_move_greedy)
        .def("best_move_temperature", &ValueLeafMCTSTree::best_move_temperature,
             py::arg("temperature"), py::arg("rng_seed") = 0u)
        // Tree-reuse interface -------------------------------------------------
        // advance_to_child(move) → bool
        //   Re-roots the tree at the child reached by `move`, preserving its
        //   entire subtree and visit statistics.  Returns True on success, False
        //   if that move was never expanded (caller should create a fresh tree).
        .def("advance_to_child", &ValueLeafMCTSTree::advance_to_child,
             py::arg("move"),
             "Re-root the tree at the child reached by 'move'. "
             "Returns True if the child exists, False otherwise.")
        // reset_to_board(board)
        //   Discard all nodes and restart from a new board position.
        //   Use as fallback when advance_to_child() returns False.
        .def("reset_to_board", &ValueLeafMCTSTree::reset_to_board,
             py::arg("board"),
             "Discard all nodes and restart the tree from 'board'.")
        // Virtual-loss helpers -------------------------------------------
        // apply_virtual_loss(node_idx)
        //   Walk from node_idx to root, incrementing vl_count at each node.
        //   Call this immediately after select_and_expand() for a non-terminal
        //   leaf, before the NN evaluation.  Subsequent selections in the same
        //   batch will see the node as provisionally less attractive.
        .def("apply_virtual_loss", &ValueLeafMCTSTree::apply_virtual_loss,
             py::arg("node_idx"))
        // undo_virtual_loss(node_idx)
        //   Walk from node_idx to root, decrementing vl_count.  Call this
        //   just before backpropagate() to cleanly replace the provisional
        //   virtual-loss effect with the real NN value.
        .def("undo_virtual_loss", &ValueLeafMCTSTree::undo_virtual_loss,
             py::arg("node_idx"))
        .def_property_readonly("node_count", [](const ValueLeafMCTSTree& t) {
            return static_cast<int>(t.nodes.size());
        })
        .def_property_readonly("root_idx", [](const ValueLeafMCTSTree& t) {
            return t.root_idx;
        })
        // node_expansion_stats()
        //   Walk every node in the current arena and return expansion statistics
        //   for all visited (visits > 0) non-terminal nodes.
        //
        //   Returns a 5-tuple:
        //     total_visited   int   — visited non-terminal nodes
        //     n_nfe           int   — not-fully-expanded subset (untried_moves > 0)
        //     n_expanded      int32[n_nfe]  — expanded child count per NFE node
        //     n_total         int32[n_nfe]  — total child count (expanded + untried)
        //     visits          int32[n_nfe]  — visit count per NFE node
        //
        //   Call this BEFORE advance_to_child(), i.e. after a full MCTS budget but
        //   before the tree is re-rooted, to capture the complete search state.
        .def("node_expansion_stats", [](const ValueLeafMCTSTree& tree) {
            int total_visited = 0;
            std::vector<int32_t> nfe_expanded, nfe_total, nfe_visits;

            for (const auto& node : tree.nodes) {
                if (node.visits <= 0) continue;
                if (node.is_terminal()) continue;
                int n_exp  = static_cast<int32_t>(node.child_indices.size());
                int n_untr = static_cast<int32_t>(node.untried_moves.size());
                // Guard: skip nodes with no move information (shouldn't occur)
                if (n_exp + n_untr == 0) continue;
                ++total_visited;
                if (n_untr > 0) {
                    nfe_expanded.push_back(static_cast<int32_t>(n_exp));
                    nfe_total.push_back(static_cast<int32_t>(n_exp + n_untr));
                    nfe_visits.push_back(static_cast<int32_t>(node.visits));
                }
            }

            int k = static_cast<int>(nfe_expanded.size());
            py::array_t<int32_t> exp_arr(k), tot_arr(k), vis_arr(k);
            if (k > 0) {
                std::memcpy(exp_arr.mutable_data(), nfe_expanded.data(), k * sizeof(int32_t));
                std::memcpy(tot_arr.mutable_data(), nfe_total.data(),    k * sizeof(int32_t));
                std::memcpy(vis_arr.mutable_data(), nfe_visits.data(),   k * sizeof(int32_t));
            }
            return py::make_tuple(total_visited, k, exp_arr, tot_arr, vis_arr);
        }, "Return expansion stats for all visited non-terminal nodes in the tree.");

    // -- ABSearchBot --
    py::class_<ABSearchBot>(m, "ABSearchBot")
        .def(py::init<int>(), py::arg("max_depth") = 5)
        .def("request_move", &ABSearchBot::request_move,
             py::arg("board"), py::arg("time_limit_secs") = -1.0)
        .def("request_move_with_eval", [](ABSearchBot& bot, BoardState board) {
            auto [eval_val, move] = bot.request_move_with_eval(board);
            return py::make_tuple(eval_val, move);
        }, py::arg("board"),
           "Return (eval_red, best_move) from a fixed-depth search.")
        .def_readwrite("max_depth", &ABSearchBot::max_depth)
        .def_readwrite("expansions", &ABSearchBot::expansions);

    // -- MCTSBot --
    py::class_<MCTSBot>(m, "MCTSBot")
        .def(py::init<int, int, double, double, int>(),
             py::arg("time_ms") = 1000,
             py::arg("threads") = 4,
             py::arg("exploration") = 700.0,
             py::arg("temperature") = 0.7,
             py::arg("max_rollout") = 200)
        .def("request_move", &MCTSBot::request_move,
             py::arg("board"))
        .def_readwrite("time_limit_ms", &MCTSBot::time_limit_ms)
        .def_readwrite("num_threads", &MCTSBot::num_threads)
        .def_readwrite("exploration", &MCTSBot::exploration)
        .def_readwrite("temperature", &MCTSBot::temperature)
        .def_property_readonly("nodes_created", &MCTSBot::get_nodes_created)
        .def_property_readonly("rollouts_evaluated", &MCTSBot::get_rollouts_evaluated);

    // -- Utility functions --
    m.def("from_python_board", &from_python_board,
          py::arg("py_board"),
          "Convert a Python Boardstate object to a C++ BoardState");

    m.def("encode_board", [](const BoardState& b) {
        auto enc = encode_for_nn(b);
        py::array_t<float> board_arr({4, 5, 5});
        py::array_t<float> cards_arr({3, 16});
        std::memcpy(board_arr.mutable_data(), &enc.board, 100 * sizeof(float));
        std::memcpy(cards_arr.mutable_data(), &enc.cards, 48 * sizeof(float));
        return py::make_tuple(board_arr, cards_arr);
    }, py::arg("board"), "Encode a board state for NN evaluation");

    m.def("card_name", [](int idx) -> std::string {
        if (idx < 0 || idx >= 16) throw std::out_of_range("Card index out of range");
        return std::string(DECK[idx].name);
    }, py::arg("idx"), "Get card name from index");

    m.def("card_index", [](const std::string& name) {
        return card_name_to_index(name);
    }, py::arg("name"), "Get card index from name");

    m.def("set_mcts_python_compat", [](bool enabled) {
        MCTS_PYTHON_COMPAT = enabled;
    }, py::arg("enabled"),
       "Enable/disable Python-compat mode for C++ MCTS (single-thread when true)");

    m.def("get_mcts_python_compat", []() {
        return MCTS_PYTHON_COMPAT;
    }, "Get current Python-compat mode flag for C++ MCTS");

    // Batch build + evaluate: given a list of BoardStates and max_depth,
    // builds trees, returns leaf tensors for batched NN eval.
    // This is the key function for training speedup.
    m.def("build_minimax_trees", [](
        const std::vector<BoardState>& boards,
        int max_depth
    ) {
        // Build all trees
        std::vector<MinimaxTree> trees;
        trees.reserve(boards.size());
        for (const auto& b : boards) {
            trees.push_back(MinimaxTree::build(b, max_depth));
        }

        // Gather all leaf tensors
        int total_leaves = 0;
        std::vector<int> tree_leaf_counts;
        tree_leaf_counts.reserve(trees.size());
        for (const auto& t : trees) {
            int n = static_cast<int>(t.leaf_indices.size());
            tree_leaf_counts.push_back(n);
            total_leaves += n;
        }

        py::array_t<float> all_boards({total_leaves, 4, 5, 5});
        py::array_t<float> all_cards({total_leaves, 3, 16});
        float* b_ptr = all_boards.mutable_data();
        float* c_ptr = all_cards.mutable_data();

        int offset = 0;
        for (const auto& t : trees) {
            auto lt = t.get_leaf_tensors();
            if (lt.count > 0) {
                std::memcpy(b_ptr + offset * 100, lt.boards.data(),
                            lt.count * 100 * sizeof(float));
                std::memcpy(c_ptr + offset * 48, lt.cards.data(),
                            lt.count * 48 * sizeof(float));
            }
            offset += lt.count;
        }

        // Return trees as opaque list + numpy arrays + counts
        py::list tree_list;
        for (auto& t : trees) {
            tree_list.append(py::cast(std::move(t)));
        }

        return py::make_tuple(tree_list, all_boards, all_cards,
                              py::cast(tree_leaf_counts));
    }, py::arg("boards"), py::arg("max_depth"),
       "Build minimax trees for multiple boards and return batched leaf tensors");

    m.def("set_values_and_backup", [](
        py::list tree_list,
        py::array_t<float> all_values,
        const std::vector<int>& leaf_counts
    ) {
        const float* v_ptr = all_values.data();
        int offset = 0;
        for (size_t i = 0; i < leaf_counts.size(); ++i) {
            auto& tree = tree_list[i].cast<MinimaxTree&>();
            tree.set_leaf_values(v_ptr + offset, leaf_counts[i]);
            tree.backup();
            offset += leaf_counts[i];
        }
    }, py::arg("trees"), py::arg("values"), py::arg("leaf_counts"),
       "Set NN values for all tree leaves and run negamax backup");
}
