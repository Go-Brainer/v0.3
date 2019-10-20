import dlgo.agent.naive as naive
from dlgo.agent.human_agent import HumanAgent
from dlgo.minimax.depthprune import DepthPrunedAgent
from dlgo import goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move
from os import name, system
from sys import argv
import time
import argparse


def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')


def capture_diff(game_state):
    black_stones = 0
    white_stones = 0
    for r in range(1, game_state.board.num_rows + 1):
        for c in range(1, game_state.board.num_cols + 1):
            p = gotypes.Point(r, c)
            color = game_state.board.get(p)
            if color == gotypes.Player.black:
                black_stones += 1
            elif color == gotypes.Player.white:
                white_stones += 1
    diff = black_stones - white_stones                    # <1>
    if game_state.next_player == gotypes.Player.black:    # <2>
        return diff                                       # <2>
    return -1 * diff


def get_options(args=argv[1:]):
    parser = argparse.ArgumentParser(prog="Go-Brainer", description="An AI Go Bot")

    parser.add_argument('b_agent', type=lambda c: c.lower(), choices=['r', 'd', 'h'],
                        help="Choose Player Agent for black: r - random, d - depth pruned minimax, h - human")
    parser.add_argument('w_agent', type=lambda c: c.lower(), choices=['r', 'd', 'h'],
                        help="Choose Player Agent for white: r - random, d - depth pruned minimax, h - human")
    parser.add_argument('-s', '--size', type=int, default=5, help="Board size is n by n. Must be between 5 and 19.")
    parser.add_argument('-k', '--komi', type=float, default=7.5, help="Sets komi. Must be between 0 and 10.")

    options = parser.parse_args(args)
    return options


def main():
    options = get_options(argv[1:])
    size = abs(options.size)
    komi = abs(options.komi)

    if size < 5 or size > 19:
        print("Board size must be between 5 and 19")
        exit(-1)

    if komi < 0 or komi > 10:
        print("Komi must be between 0 and 10")
        exit(-1)

    game = goboard.GameState.new_game(size)
    if options.b_agent == 'r':
        b_agent = naive.RandomBot()
    elif options.b_agent == 'd':
        b_agent = DepthPrunedAgent(3, capture_diff)
    elif options.b_agent == 'h':
        b_agent = HumanAgent()
    else:
        b_agent = None
        print(options)
        print("Invalid options error")
        exit(0)

    if options.w_agent == 'r':
        w_agent = naive.RandomBot()
    elif options.w_agent == 'd':
        w_agent = DepthPrunedAgent(3, capture_diff)
    elif options.w_agent == 'h':
        w_agent = HumanAgent()
    else:
        w_agent = None
        print(options)
        print("Invalid options error")
        exit(0)

    players = {
        gotypes.Player.black: b_agent,
        gotypes.Player.white: w_agent
    }

    print_board(game.board)

    while not game.is_over():
        # Since making a play is necessary for changing board state but also changes
        # next player we must save the current player
        player_before = game.next_player
        move = players[game.next_player].select_move(game)
        clear()
        game = game.apply_move(move)
        print_board(game.board)
        time.sleep(.1)
        print_move(player_before, move)
    # end of main game loop

    game.print_game_results(komi)


if __name__ == '__main__':
    main()
