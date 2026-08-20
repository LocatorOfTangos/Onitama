#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using Coord = std::pair<int, int>;

static std::mt19937& thread_rng() {
    static thread_local std::mt19937 rng(std::random_device{}());
    return rng;
}

struct Card {
    std::string name;
    int idx;
    std::vector<Coord> moveset;

    Card() : idx(-1) {}
    Card(const std::string& n, int i, const std::vector<Coord>& ms) : name(n), idx(i), moveset(ms) {}
};

struct Deck {
    static const std::vector<Card>& get_deck() {
        static const std::vector<Card> deck = {
            Card("Tiger", 0, {{0, 2}, {0, -1}}),
            Card("Monkey", 1, {{1, 1}, {-1, 1}, {-1, -1}, {1, -1}}),
            Card("Dragon", 2, {{2, 1}, {-2, 1}, {-1, -1}, {1, -1}}),
            Card("Crab", 3, {{0, 1}, {2, 0}, {-2, 0}}),
            Card("Mantis", 4, {{1, 1}, {-1, 1}, {0, -1}}),
            Card("Frog", 5, {{-1, 1}, {-2, 0}, {1, -1}}),
            Card("Elephant", 6, {{-1, 1}, {1, 1}, {-1, 0}, {1, 0}}),
            Card("Rooster", 7, {{1, 1}, {-1, 0}, {1, 0}, {-1, -1}}),
            Card("Boar", 8, {{0, 1}, {-1, 0}, {1, 0}}),
            Card("Ox", 9, {{0, 1}, {1, 0}, {0, -1}}),
            Card("Crane", 10, {{0, 1}, {1, -1}, {-1, -1}}),
            Card("Eel", 11, {{-1, 1}, {1, 0}, {-1, -1}}),
            Card("Horse", 12, {{0, 1}, {-1, 0}, {0, -1}}),
            Card("Cobra", 13, {{1, 1}, {-1, 0}, {1, -1}}),
            Card("Goose", 14, {{-1, 1}, {-1, 0}, {1, 0}, {1, -1}}),
            Card("Rabbit", 15, {{1, 1}, {2, 0}, {-1, -1}}),
        };
        return deck;
    }

    static const Card* card_by_idx(int idx) {
        const auto& d = get_deck();
        for (const auto& c : d) {
            if (c.idx == idx) return &c;
        }
        return nullptr;
    }
};

struct Move {
    Coord start;
    Coord dest;
    int card_idx;

    Move() : start({-1, -1}), dest({-1, -1}), card_idx(-1) {}
    Move(Coord s, Coord d, int c) : start(s), dest(d), card_idx(c) {}
};

class BoardState {
public:
    struct UndoRecord {
        int prev_turn;
        std::array<const Card*, 2> prev_red_cards;
        std::array<const Card*, 2> prev_blue_cards;
        const Card* prev_trans;
        int captured_index;  // -1 if no capture
        int start_index;     // index of moved piece
        Move move;
    };

    std::array<std::optional<Coord>, 10> positions;
    std::array<const Card*, 2> red_cards;
    std::array<const Card*, 2> blue_cards;
    const Card* trans_card;
    int turn_num;
    bool red_start;

    BoardState() : turn_num(0), red_start(true), trans_card(nullptr) {
        positions[0] = Coord{2, 0};
        positions[1] = Coord{0, 0};
        positions[2] = Coord{1, 0};
        positions[3] = Coord{3, 0};
        positions[4] = Coord{4, 0};
        positions[5] = Coord{2, 4};
        positions[6] = Coord{0, 4};
        positions[7] = Coord{1, 4};
        positions[8] = Coord{3, 4};
        positions[9] = Coord{4, 4};

        const auto& deck = Deck::get_deck();
        std::vector<int> indices(deck.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), thread_rng());
        red_cards = {&deck[indices[0]], &deck[indices[1]]};
        blue_cards = {&deck[indices[2]], &deck[indices[3]]};
        trans_card = &deck[indices[4]];
        red_start = (thread_rng()() % 2 == 0);
    }

    BoardState(const std::array<std::optional<Coord>, 10>& pos,
               const std::array<const Card*, 2>& r_cards,
               const std::array<const Card*, 2>& b_cards,
               const Card* t_card,
               int turn,
               bool r_start)
        : positions(pos), red_cards(r_cards), blue_cards(b_cards),
          trans_card(t_card), turn_num(turn), red_start(r_start) {}

    std::shared_ptr<BoardState> clone() const {
        return std::make_shared<BoardState>(*this);
    }

    std::string turn_colour(bool opnt_colour = false) const {
        bool turn_odd = (turn_num % 2 != 0);
        bool rhs = opnt_colour ? !turn_odd : turn_odd;
        return (red_start ^ rhs) ? "RED" : "BLUE";
    }

    std::optional<std::pair<std::string, std::string>> is_won() const {
        if (positions[0] && *positions[0] == Coord{2, 4}) {
            return std::make_pair("RED", "STREAM");
        }
        if (positions[5] && *positions[5] == Coord{2, 0}) {
            return std::make_pair("BLUE", "STREAM");
        }
        if (!positions[0]) {
            return std::make_pair("BLUE", "STONE");
        }
        if (!positions[5]) {
            return std::make_pair("RED", "STONE");
        }
        return std::nullopt;
    }

    bool is_terminal() const {
        return is_won().has_value();
    }

    std::vector<Move> possible_moves() const {
        bool red_turn = (turn_colour() == "RED");
        int mul = red_turn ? 1 : -1;

        const auto& cards = red_turn ? red_cards : blue_cards;
        int start_idx = red_turn ? 0 : 5;
        int end_idx = red_turn ? 5 : 10;

        std::vector<Coord> own_pieces;
        own_pieces.reserve(5);
        for (int i = start_idx; i < end_idx; ++i) {
            if (positions[i]) {
                own_pieces.push_back(*positions[i]);
            }
        }

        std::vector<Move> moves;
        moves.reserve(40);

        auto add_moves_from_card = [&](const Card* card, const Coord& position) {
            for (const Coord& card_move : card->moveset) {
                Coord destination = {
                    position.first - mul * card_move.first,
                    position.second + mul * card_move.second,
                };
                if (destination.first < 0 || destination.first > 4) continue;
                if (destination.second < 0 || destination.second > 4) continue;
                if (std::find(own_pieces.begin(), own_pieces.end(), destination) != own_pieces.end()) continue;
                moves.emplace_back(position, destination, card->idx);
            }
        };

        for (const Coord& piece_pos : own_pieces) {
            add_moves_from_card(cards[0], piece_pos);
            add_moves_from_card(cards[1], piece_pos);
        }

        std::shuffle(moves.begin(), moves.end(), thread_rng());

        return moves;
    }

    std::vector<Move> possible_moves_search_optimised() const {
        bool red_turn = (turn_colour() == "RED");
        int mul = red_turn ? 1 : -1;

        const auto& cards = red_turn ? red_cards : blue_cards;
        int own_start = red_turn ? 0 : 5;
        int own_end = red_turn ? 5 : 10;
        int opp_start = red_turn ? 5 : 0;
        int opp_end = red_turn ? 10 : 5;
        Coord stream_target = red_turn ? Coord{2, 4} : Coord{2, 0};

        std::vector<Coord> own_positions;
        own_positions.reserve(5);
        for (int i = own_start; i < own_end; ++i) {
            if (positions[i]) own_positions.push_back(*positions[i]);
        }

        std::vector<Coord> opp_positions;
        opp_positions.reserve(5);
        for (int i = opp_start; i < opp_end; ++i) {
            if (positions[i]) opp_positions.push_back(*positions[i]);
        }

        std::vector<Move> captures;
        std::vector<Move> other_moves;
        captures.reserve(20);
        other_moves.reserve(20);

        for (int i = own_start; i < own_end; ++i) {
            if (!positions[i]) continue;
            const Coord position = *positions[i];
            bool is_master = (i == own_start);

            for (const Card* card : cards) {
                for (const Coord& card_move : card->moveset) {
                    Coord destination = {
                        position.first - mul * card_move.first,
                        position.second + mul * card_move.second,
                    };

                    if (destination.first < 0 || destination.first > 4) continue;
                    if (destination.second < 0 || destination.second > 4) continue;
                    if (std::find(own_positions.begin(), own_positions.end(), destination) != own_positions.end()) continue;

                    Move move(position, destination, card->idx);

                    // Prioritize immediate win by capturing opponent master.
                    if (positions[opp_start] && destination == *positions[opp_start]) {
                        return {move};
                    }

                    // Prioritize immediate stream win by moving own master onto temple.
                    if (is_master && destination == stream_target) {
                        return {move};
                    }

                    if (std::find(opp_positions.begin(), opp_positions.end(), destination) != opp_positions.end()) {
                        captures.push_back(move);
                    } else {
                        other_moves.push_back(move);
                    }
                }
            }
        }

        captures.insert(captures.end(), other_moves.begin(), other_moves.end());
        return captures;
    }

    UndoRecord apply_move(const Move& move) {
        UndoRecord undo;
        undo.prev_turn = turn_num;
        undo.prev_red_cards = red_cards;
        undo.prev_blue_cards = blue_cards;
        undo.prev_trans = trans_card;
        undo.move = move;

        bool red_turn = (turn_colour() == "RED");
        auto& cards = red_turn ? red_cards : blue_cards;
        if (cards[0]->idx == move.card_idx) {
            const Card* used = cards[0];
            cards[0] = trans_card;
            trans_card = used;
        } else {
            const Card* used = cards[1];
            cards[1] = trans_card;
            trans_card = used;
        }

        undo.captured_index = -1;
        for (int i = 0; i < 10; ++i) {
            if (positions[i] && *positions[i] == move.dest) {
                undo.captured_index = i;
                positions[i] = std::nullopt;
                break;
            }
        }

        undo.start_index = -1;
        for (int i = 0; i < 10; ++i) {
            if (positions[i] && *positions[i] == move.start) {
                undo.start_index = i;
                positions[i] = move.dest;
                break;
            }
        }

        turn_num += 1;
        return undo;
    }

    void undo_move(const UndoRecord& undo) {
        // Restore moved piece
        if (undo.start_index >= 0) {
            positions[undo.start_index] = undo.move.start;
        }
        // Restore captured piece
        if (undo.captured_index >= 0) {
            positions[undo.captured_index] = undo.move.dest;
        }
        // Restore cards and turn
        red_cards = undo.prev_red_cards;
        blue_cards = undo.prev_blue_cards;
        trans_card = undo.prev_trans;
        turn_num = undo.prev_turn;
    }

    int evaluate(int depth = 0) const {
        static const int CENTER_PRIORITY[5][5] = {
            {-10, 0, 10, 0, -10},
            {0, 10, 20, 10, 0},
            {10, 20, 30, 20, 10},
            {0, 10, 20, 10, 0},
            {-10, 0, 10, 0, -10},
        };
        static const int OPENING_MASTER_POSITIONAL_VALUE[5][5] = {
            {0, 10, -20, -60, -100},
            {0, 10, -20, -60, -100},
            {0, 10, -20, -60, -100},
            {0, 10, -20, -60, -100},
            {0, 10, -20, -60, -100},
        };
        static const int MIDGAME_MASTER_POSITIONAL_VALUE[5][5] = {
            {-20, 0, 0, -40, -80},
            {-20, 0, 0, -40, -80},
            {-20, 0, 0, -40, -80},
            {-20, 0, 0, -40, -80},
            {-20, 0, 0, -40, -80},
        };
        static const int ENDGAME_MASTER_POSITIONAL_VALUE[5][5] = {
            {-100, -60, 20, 60, 40},
            {-100, -60, 20, 80, 100},
            {-100, -60, 20, 100, 100},
            {-100, -60, 20, 80, 100},
            {-100, -60, 20, 60, 40},
        };
        static const int STUDENT_VALUE = 50;

        auto winner = is_won();
        if (winner) {
            int depth_discount = depth * 5;
            return winner->first == "RED" ? (1000 - depth_discount) : (-1000 + depth_discount);
        }

        int score = 0;

        for (int i = 1; i < 5; ++i) {
            if (positions[i]) {
                int x = positions[i]->first;
                int y = positions[i]->second;
                score += CENTER_PRIORITY[x][y] + STUDENT_VALUE;
            }
        }
        for (int i = 6; i < 10; ++i) {
            if (positions[i]) {
                int x = positions[i]->first;
                int y = positions[i]->second;
                score -= CENTER_PRIORITY[x][y] + STUDENT_VALUE;
            }
        }

        int num_red_students = 0;
        int num_blue_students = 0;
        for (int i = 1; i < 5; ++i) {
            if (positions[i]) {
                num_red_students += 1;
            }
        }
        for (int i = 6; i < 10; ++i) {
            if (positions[i]) {
                num_blue_students += 1;
            }
        }

        int min_students = std::min(num_red_students, num_blue_students);
        enum class Stage { OPENING, MIDGAME, ENDGAME };
        Stage stage = Stage::ENDGAME;
        if (min_students >= 4) {
            stage = Stage::OPENING;
        } else if (min_students >= 2) {
            stage = Stage::MIDGAME;
        }

        if (positions[0]) {
            int rx = positions[0]->first;
            int ry = positions[0]->second;
            if (stage == Stage::OPENING) {
                score += OPENING_MASTER_POSITIONAL_VALUE[rx][ry];
            } else if (stage == Stage::MIDGAME) {
                score += MIDGAME_MASTER_POSITIONAL_VALUE[rx][ry];
            } else {
                score += ENDGAME_MASTER_POSITIONAL_VALUE[rx][ry];
            }
        }

        if (positions[5]) {
            int bx = positions[5]->first;
            int by = positions[5]->second;
            if (stage == Stage::OPENING) {
                score -= OPENING_MASTER_POSITIONAL_VALUE[4 - bx][4 - by];
            } else if (stage == Stage::MIDGAME) {
                score -= MIDGAME_MASTER_POSITIONAL_VALUE[4 - bx][4 - by];
            } else {
                score -= ENDGAME_MASTER_POSITIONAL_VALUE[4 - bx][4 - by];
            }
        }

        return score;
    }
};

class MCTSNode {
public:
    std::shared_ptr<BoardState> board;
    MCTSNode* parent;
    Move move;
    std::vector<std::unique_ptr<MCTSNode>> children;
    std::vector<Move> untried_moves;
    int visits;
    double value;

    MCTSNode(std::shared_ptr<BoardState> b, MCTSNode* p = nullptr, const Move& m = Move())
        : board(std::move(b)), parent(p), move(m), visits(0), value(0.0) {
        untried_moves = board->possible_moves_search_optimised();
    }

    bool is_fully_expanded() const {
        return untried_moves.empty();
    }

    bool is_terminal() const {
        return board->is_terminal();
    }
};

class MCTSBot {
public:
    int num_threads;
    double exploration;
    double temperature;  // <= 0 means uniform random rollouts
    int time_limit_ms;
    std::atomic<int> nodes_created;
    std::atomic<int> rollouts_evaluated;
    std::mutex tree_mutex;

    MCTSBot(int threads = 4, double explore = 20.0, double temp = 0.7, int time_limit = 1000)
        : num_threads(threads), exploration(explore), temperature(temp),
          time_limit_ms(time_limit), nodes_created(0), rollouts_evaluated(0) {}

    double ucb(MCTSNode* node, double parent_visits, double mul) const {
        if (node->visits == 0) {
            return std::numeric_limits<double>::infinity();
        }
        double exploitation = mul * node->value / static_cast<double>(node->visits);
        double exploration_term = exploration * std::sqrt(std::log(std::max(1.0, parent_visits)) / node->visits);
        return exploitation + exploration_term;
    }

    MCTSNode* select(MCTSNode* root) const {
        MCTSNode* node = root;
        while (!node->is_terminal() && node->is_fully_expanded() && !node->children.empty()) {
            double parent_visits = static_cast<double>(std::max(1, node->visits));
            double mul = (node->board->turn_colour() == "RED") ? 1.0 : -1.0;
            node = (*std::max_element(
                        node->children.begin(),
                        node->children.end(),
                        [&](const std::unique_ptr<MCTSNode>& a, const std::unique_ptr<MCTSNode>& b) {
                            return ucb(a.get(), parent_visits, mul) < ucb(b.get(), parent_visits, mul);
                        }))
                       .get();
        }
        return node;
    }

    MCTSNode* expand(MCTSNode* node) {
        if (node->untried_moves.empty()) {
            return node;
        }

        Move m = node->untried_moves.back();
        node->untried_moves.pop_back();

        auto child_board = node->board->clone();
        child_board->apply_move(m);

        node->children.emplace_back(std::make_unique<MCTSNode>(child_board, node, m));
        nodes_created.fetch_add(1, std::memory_order_relaxed);
        return node->children.back().get();
    }

    void backpropagate(MCTSNode* node, double reward) {
        while (node != nullptr) {
            node->visits += 1;
            node->value += reward;
            node = node->parent;
        }
    }

    Move select_move_softmax(const std::vector<Move>& moves, const std::vector<double>& scores,
                              std::mt19937& rng) const {
        if (moves.size() == 1) return moves[0];

        double max_score = *std::max_element(scores.begin(), scores.end());

        std::vector<double> exp_scores;
        exp_scores.reserve(moves.size());
        double sum_exp = 0.0;
        for (double score : scores) {
            double e = std::exp((score - max_score) / temperature);
            exp_scores.push_back(e);
            sum_exp += e;
        }

        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(rng);
        double cumulative = 0.0;
        for (size_t i = 0; i < moves.size(); ++i) {
            cumulative += exp_scores[i] / sum_exp;
            if (r <= cumulative) return moves[i];
        }
        return moves.back();
    }

    double simulate(std::shared_ptr<BoardState> start_board) {
        BoardState sim = *start_board;
        int depth = 0;

        while (!sim.is_terminal() && depth < 200) {
            std::vector<Move> moves = sim.possible_moves_search_optimised();
            if (moves.empty()) break;

            Move selected;
            if (temperature <= 0.0) {
                // Uniform random rollout
                std::uniform_int_distribution<int> dist(0, static_cast<int>(moves.size()) - 1);
                selected = moves[dist(thread_rng())];
            } else {
                // Softmax-weighted rollout: evaluate each candidate via apply/undo
                double mul = (sim.turn_colour() == "RED") ? 1.0 : -1.0;
                std::vector<double> scores;
                scores.reserve(moves.size());
                for (const Move& m : moves) {
                    auto undo = sim.apply_move(m);
                    scores.push_back(mul * static_cast<double>(sim.evaluate()));
                    sim.undo_move(undo);
                }
                selected = select_move_softmax(moves, scores, thread_rng());
            }

            // Apply selected move forward — no undo record stored.
            // The sim board is a throwaway local copy; once we reach a
            // terminal state (or depth limit), we just evaluate and return.
            sim.apply_move(selected);
            depth += 1;
        }

        rollouts_evaluated.fetch_add(1, std::memory_order_relaxed);
        return static_cast<double>(sim.evaluate(depth));
    }

    Move search(std::shared_ptr<BoardState> root_board) {
        auto root = std::make_unique<MCTSNode>(root_board);
        nodes_created.store(1, std::memory_order_relaxed);
        rollouts_evaluated.store(0, std::memory_order_relaxed);

        auto start = std::chrono::steady_clock::now();
        auto deadline = start + std::chrono::milliseconds(time_limit_ms);

        auto worker = [&]() {
            while (std::chrono::steady_clock::now() < deadline) {
                MCTSNode* node;
                std::shared_ptr<BoardState> sim_board;
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

        std::vector<std::thread> threads;
        threads.reserve(std::max(1, num_threads));
        for (int i = 0; i < std::max(1, num_threads); ++i) {
            threads.emplace_back(worker);
        }
        for (auto& t : threads) {
            t.join();
        }

        MCTSNode* best = nullptr;
        int best_visits = -1;
        for (auto& child : root->children) {
            if (child->visits > best_visits) {
                best_visits = child->visits;
                best = child.get();
            }
        }

        if (best != nullptr) {
            return best->move;
        }

        auto fallback = root_board->possible_moves_search_optimised();
        if (!fallback.empty()) {
            return fallback.front();
        }
        return Move();
    }
};

int main() {
    std::array<std::optional<Coord>, 10> positions;
    for (int i = 0; i < 10; ++i) {
        int x, y;
        std::cin >> x >> y;
        if (x == -1 && y == -1) {
            positions[i] = std::nullopt;
        } else {
            positions[i] = Coord{x, y};
        }
    }

    int red_card1_idx, red_card2_idx, blue_card1_idx, blue_card2_idx, trans_card_idx;
    std::cin >> red_card1_idx >> red_card2_idx >> blue_card1_idx >> blue_card2_idx >> trans_card_idx;

    int turn_num, red_start_int, time_limit_ms, num_threads;
    std::cin >> turn_num >> red_start_int >> time_limit_ms >> num_threads;

    double exploration_val, temperature_val;
    std::cin >> exploration_val >> temperature_val;

    bool red_start = (red_start_int != 0);

    auto board = std::make_shared<BoardState>(
        positions,
        std::array<const Card*, 2>{Deck::card_by_idx(red_card1_idx), Deck::card_by_idx(red_card2_idx)},
        std::array<const Card*, 2>{Deck::card_by_idx(blue_card1_idx), Deck::card_by_idx(blue_card2_idx)},
        Deck::card_by_idx(trans_card_idx),
        turn_num,
        red_start
    );

    MCTSBot bot(num_threads, exploration_val, temperature_val, time_limit_ms);
    Move best_move = bot.search(board);

    std::cout << best_move.start.first << " " << best_move.start.second << " "
              << best_move.dest.first << " " << best_move.dest.second << " "
              << best_move.card_idx << std::endl;

    return 0;
}
