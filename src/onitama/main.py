import random
import re
import time
from onitama import Boardstate
from onitama.opponents import random_bot, shortsighted_bot, ab_search_bot, advanced_search_bot, mcts_bot, mcts_vn131
from onitama.opponents.pmcts import pmcts_bot
from onitama.engine.onitama_engine import (
    ABSearchBot as CppABBot,
    MCTSBot as CppMCTSBot,
    from_python_board,
)


def request_move(board):
    while True:
        move_input = input("Move a piece: ")
        if not re.match(r"[a-z]\d[a-z]\d", move_input):
            print("Could not parse input.")
            continue

        raw_move_coords = []
        for num, char in enumerate(move_input):
            if num % 2 == 0:
                raw_move_coords.append(101 - ord(char))
            else:
                raw_move_coords.append(int(char))

        if len(raw_move_coords) != 4 or any(x not in range(5) for x in raw_move_coords):
            print("Invalid Coordinates.")
            continue

        move_coords = [(raw_move_coords[0], raw_move_coords[1]), (raw_move_coords[2], raw_move_coords[3])]
        move = board.validate_move(move_coords)
        if move:
            return move


def print_victory(victory_type):
    string = f"{victory_type[0]} wins by way of the {victory_type[1]}"
    print("\n" + len(string) * "=" + f"\n{string}\n" + len(string) * "=" + "\n")
    return


class CppBotWrapper:
    """Wraps a C++ bot so it can be used with the Python Boardstate game loop."""
    def __init__(self, cpp_bot, name, use_time_limit=False):
        self.cpp_bot = cpp_bot
        self.name = name
        self.use_time_limit = use_time_limit

    def request_move(self, py_board, time_limit=None):
        cpp_board = from_python_board(py_board)
        if self.use_time_limit and time_limit is not None:
            cpp_move = self.cpp_bot.request_move(cpp_board, time_limit)
        else:
            cpp_move = self.cpp_bot.request_move(cpp_board)
        return cpp_move.to_coords()


def main(tl = None):
    time_limit = 1.0 if tl is None else tl
    print("\nWelcome to ONITAMA\n")
    response = input("Enter 1 to play locally, 2 - random bot, 3 - shortsighted bot,\n"
                     "4 - simple search bot, 5 - MCTS bot, 6 - PMCTS, 7 - C-ABSearch,\n"
                     "8 - C-MCTS, 9 - MCTS-VN131, 0 - watch an exhibition match: ")

    if response == '1':
        # two player local
        board = Boardstate.Boardstate()
        while True:
            print()
            print(board.board_str())
            victory_type = board.is_won()
            if victory_type:
                print_victory(victory_type)
                return
            print(f"{board.turn_colour()}\'s turn.")
            move = request_move(board)
            board.execute_move(move)

    elif response == '2':
        player_colour = "RED" if random.randint(0,1) == 0 else "BLUE"
        bot = random_bot.RandomBot()
        board = Boardstate.Boardstate()
        print(f"You are {player_colour}.")
        while True:
            print()
            print(board.board_str())
            victory_type = board.is_won()
            if victory_type:
                print_victory(victory_type)
                return
            print(f"{board.turn_colour()}\'s turn.")
            if board.turn_colour() == player_colour:
                move = request_move(board)
            else:
                move = bot.request_move(board)
                print('Bot plays: ', chr(101 - move[0][0]), move[0][1], chr(101 - move[1][0]), move[1][1], sep='')
            board.execute_move(move)

    elif response == '3':
        player_colour = "BLUE"
        bot = shortsighted_bot.ShortSightedBot()
        board = Boardstate.Boardstate()
        print("You are BLUE.")
        while True:
            print()
            print(board.board_str())
            victory_type = board.is_won()
            if victory_type:
                print_victory(victory_type)
                return
            print(f"{board.turn_colour()}\'s turn.")
            if board.turn_colour() == player_colour:
                move = request_move(board)
            else:
                move = bot.request_move(board)
                print('Bot plays: ', chr(101 - move[0][0]), move[0][1], chr(101 - move[1][0]), move[1][1], sep='')
            board.execute_move(move)

    elif response == '4':
        player_colour = "RED" if random.randint(0,1) == 0 else "BLUE"
        bot = ab_search_bot.ABSearchBot()
        board = Boardstate.Boardstate()
        print(f"You are {player_colour}.")
        while True:
            print()
            print(board.board_str())
            victory_type = board.is_won()
            if victory_type:
                print_victory(victory_type)
                return
            print(f"{board.turn_colour()}\'s turn.\n")
            if board.turn_colour() == player_colour:
                move = request_move(board)
            else:
                move = bot.request_move(board, time_limit=time_limit)
                print('Bot plays: ', chr(101 - move[0][0]), move[0][1], chr(101 - move[1][0]), move[1][1], sep='')
            board.execute_move(move)

    elif response == '5':
        player_colour = "BLUE"
        bot = mcts_bot.MCTSBot()
        board = Boardstate.Boardstate()
        print("You are BLUE.")
        while True:
            print()
            print(board.board_str())
            victory_type = board.is_won()
            if victory_type:
                print_victory(victory_type)
                return
            print(f"{board.turn_colour()}\'s turn.")
            if board.turn_colour() == player_colour:
                move = request_move(board)
            else:
                move = bot.request_move(board, time_limit=time_limit)
                print('Bot plays: ', chr(101 - move[0][0]), move[0][1], chr(101 - move[1][0]), move[1][1], sep='')
            board.execute_move(move)

    elif response == '0':
        # exhibition: select which bots play each other
        print("\nSelect bot type for RED player: 1=Random, 2=ShortSighted, 3=ABSearch, 4=AdvancedSearch, 5=MCTS, 6=PMCTS, 7=C-ABSearch, 8=C-MCTS, 9=MCTS-VN131")
        red_choice = input("Enter 1-9: ").strip()
        print("\nSelect bot type for BLUE player: 1=Random, 2=ShortSighted, 3=ABSearch, 4=AdvancedSearch, 5=MCTS, 6=PMCTS, 7=C-ABSearch, 8=C-MCTS, 9=MCTS-VN131")
        blue_choice = input("Enter 1-9: ").strip()
        
        bot_map = {
            '1': ('Random', random_bot.RandomBot()),
            '2': ('ShortSighted', shortsighted_bot.ShortSightedBot()),
            '3': ('ABSearch', ab_search_bot.ABSearchBot()),
            '4': ('AdvancedSearch', advanced_search_bot.AdvancedSearchBot()),
            '5': ('MCTS', mcts_bot.MCTSBot()),
            '6': ('PMCTS', pmcts_bot.PMCTSBot()),
            '7': ('C-ABSearch', CppBotWrapper(CppABBot(5), 'C-ABSearch', use_time_limit=True)),
            '8': ('C-MCTS', CppBotWrapper(CppMCTSBot(1000), 'C-MCTS')),
            '9': ('MCTS-VN131', mcts_vn131.MCTS_VN131())
        }
        
        red_name, red_bot = bot_map.get(red_choice, ('ShortSighted', shortsighted_bot.ShortSightedBot()))
        blue_name, blue_bot = bot_map.get(blue_choice, ('ABSearch', ab_search_bot.ABSearchBot()))
        
        print(f"\nExhibition: {red_name} (RED) vs {blue_name} (BLUE)\n")
        
        board = Boardstate.Boardstate()
        while True:
            print()
            print(board.board_str('RED'))
            victory_type = board.is_won()
            if victory_type:
                print_victory(victory_type)
                return
            print(f"{board.turn_colour()}\'s turn.\n")

            if board.turn_colour() == "RED":
                start_time = time.time()
                move = red_bot.request_move(board, time_limit=time_limit)
                elapsed = time.time() - start_time
                bot_name = red_name
            else:
                start_time = time.time()
                move = blue_bot.request_move(board, time_limit=time_limit)
                elapsed = time.time() - start_time
                bot_name = blue_name

            remaining = 1.0 - elapsed
            if remaining > 0:
                time.sleep(remaining)

            print(f"{bot_name} ({board.turn_colour()}) bot plays {chr(101-move[0][0])}{move[0][1]}{chr(101-move[1][0])}{move[1][1]}\n")
            board.execute_move(move)

    elif response == '6':
        player_colour = "BLUE"
        bot = pmcts_bot.PMCTSBot()
        board = Boardstate.Boardstate()
        print("You are BLUE.")
        while True:
            print()
            print(board.board_str())
            victory_type = board.is_won()
            if victory_type:
                print_victory(victory_type)
                return
            print(f"{board.turn_colour()}\'s turn.")
            if board.turn_colour() == player_colour:
                move = request_move(board)
            else:
                move = bot.request_move(board, time_limit=time_limit)
                print('Bot plays: ', chr(101 - move[0][0]), move[0][1], chr(101 - move[1][0]), move[1][1], sep='')
            board.execute_move(move)

    elif response == '7':
        player_colour = "RED" if random.randint(0,1) == 0 else "BLUE"
        bot = CppBotWrapper(CppABBot(5), 'C-ABSearch', use_time_limit=True)
        board = Boardstate.Boardstate()
        print(f"You are {player_colour}.")
        while True:
            print()
            print(board.board_str())
            victory_type = board.is_won()
            if victory_type:
                print_victory(victory_type)
                return
            print(f"{board.turn_colour()}\'s turn.")
            if board.turn_colour() == player_colour:
                move = request_move(board)
            else:
                move = bot.request_move(board, time_limit=time_limit)
                print('Bot plays: ', chr(101 - move[0][0]), move[0][1], chr(101 - move[1][0]), move[1][1], sep='')
            board.execute_move(move)

    elif response == '8':
        player_colour = "RED" if random.randint(0,1) == 0 else "BLUE"
        bot = CppBotWrapper(CppMCTSBot(time_limit*1000), 'C-MCTS')
        board = Boardstate.Boardstate()
        print(f"You are {player_colour}.")
        while True:
            print()
            print(board.board_str())
            victory_type = board.is_won()
            if victory_type:
                print_victory(victory_type)
                return
            print(f"{board.turn_colour()}\'s turn.")
            if board.turn_colour() == player_colour:
                move = request_move(board)
            else:
                move = bot.request_move(board)
                print('Bot plays: ', chr(101 - move[0][0]), move[0][1], chr(101 - move[1][0]), move[1][1], sep='')
            board.execute_move(move)

    
    elif response == '9':
        player_colour = "BLUE"
        bot = mcts_vn131.MCTS_VN131()
        board = Boardstate.Boardstate()
        print("You are BLUE.")
        while True:
            print()
            print(board.board_str())
            victory_type = board.is_won()
            if victory_type:
                print_victory(victory_type)
                return
            print(f"{board.turn_colour()}\'s turn.")
            if board.turn_colour() == player_colour:
                move = request_move(board)
            else:
                move = bot.request_move(board, time_limit=time_limit)
                print('Bot plays: ', chr(101 - move[0][0]), move[0][1], chr(101 - move[1][0]), move[1][1], sep='')
            board.execute_move(move)


if __name__ == "__main__":
    from sys import argv
    tl = None
    if len(argv) > 2 and argv[1] == "-tl":
        tl = float(argv[2])
    main(tl)