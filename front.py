from back import *

def start():
    print("\nWelcome to ONITAMA\n")
    two_player()

def two_player():
    print("Two player mode.\n")
    print("Usage: input move by typing start and end coordinates: \'a1b2\'\nYou may be asked to specify a card. In this case, name the desired card: \'pidgeon\'\n")
    two_player_mode()


if __name__ == "__main__":
    start()