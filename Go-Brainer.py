import dlgo.agent.naive as naive
from dlgo import goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move, point_from_coords
from six.moves import input
from os import name, system
from re import match
from sys import argv
import time
import argparse


def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')


def get_options(args=argv[1:]):
    parser = argparse.ArgumentParser(prog="Go-Brainer", description="An AI Go Bot")

    parser.add_argument('-s', '--size', type=int, default=9, help="Board size is n by n. Must be between 5 and 19.")
    parser.add_argument('-m', '--human', type=lambda c: c.lower(), choices=['b', 'w'],
                        help="Enables human v. bot mode and sets human's color.")
    parser.add_argument('-k', '--komi', type=float, default=7.5, help="Sets komi. Must be between 0 and 10.")

    options = parser.parse_args(args)
    return options


def main():
    options = get_options(argv[1:])
    size = abs(options.size)
    human = options.human
    komi = abs(options.komi)

    if size < 5 or size > 19:
        print("Board size must be between 5 and 19")
        exit(-1)

    if komi < 0 or komi > 10:
        print("Komi must be between 0 and 10")
        exit(-1)

    game = goboard.GameState.new_game(size)
    players = {}
    if human is None:
        players = {
            gotypes.Player.black: naive.RandomBot(),
            gotypes.Player.white: naive.RandomBot()
        }
    elif human == 'b':

        players = {
            gotypes.Player.black: human,
            gotypes.Player.white: naive.RandomBot()
        }
    elif human == 'w':
        players = {
            gotypes.Player.black: naive.RandomBot(),
            gotypes.Player.white: human
        }
    else:
        print(options)
        print("Invalid options error")
        exit(0)

    print_board(game.board)

    while not game.is_over():
        # Since making a play is necessary for changing board state but also changes
        # next player we must save the current player
        player_before = game.next_player
        move = None
        if players[game.next_player] == human:
            valid_move = False
            while not valid_move:
                try:
                    human_move = input('-- ').upper()
                    if match("P(ASS)*$", human_move):
                        move = goboard.Move.pass_turn()
                    elif match("R(ESIGN)*$", human_move):
                        move = goboard.Move.resign()
                    else:
                        point = point_from_coords(human_move.strip())
                        move = goboard.Move.play(point)

                    valid_move = game.is_valid_move(move)
                    if not valid_move:
                        print("Invalid move")

                except AssertionError:
                    print("Invalid move")
                except ValueError:
                    print("Invalid move")
                except IndexError:
                    print("Invalid move")
            # end of human input loop
        else:
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
