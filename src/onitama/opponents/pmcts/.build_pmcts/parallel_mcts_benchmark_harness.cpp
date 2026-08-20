#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#define main pmcts_bot_original_main
#include "/home/redmond/Projects/Onitama/opponents/pmcts/pmcts_bot.cpp"
#undef main

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: harness <threads> <time_ms>\n";
        return 2;
    }

    int threads = std::stoi(argv[1]);
    int time_ms = std::stoi(argv[2]);

    try {
        MCTSBot bot(threads, 20.0, time_ms);
        auto board = std::make_shared<BoardState>();

        auto start = std::chrono::steady_clock::now();
        Move move = bot.search(board);
        auto end = std::chrono::steady_clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();
        int nodes = bot.nodes_created.load();
        int rollouts = bot.rollouts_evaluated.load();

        std::cout << "elapsed_s=" << std::fixed << std::setprecision(6) << elapsed << "\n";
        std::cout << "nodes_created=" << nodes << "\n";
        std::cout << "rollouts_evaluated=" << rollouts << "\n";
        std::cout << "move_card_idx=" << move.card_idx << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "runtime_error=" << ex.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "runtime_error=unknown\n";
        return 1;
    }
}
