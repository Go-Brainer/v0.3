import dlgo.agent.naive as naive
from dlgo import goboard_slow as goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move, point_from_coords
from six.moves import input
from os import name, system
from dlgo.scoring import compute_game_result
from re import match
from sys import argv
import time


def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')


def print_usage():
    print("Usage: python3 Go-Brainer.py <5-19> <hb|bh|bb>")


def main():
    if len(argv) is not 3:
        print_usage()
        exit(0)
    if not argv[1].isnumeric() or \
            int(argv[1]) < 5 or \
            int(argv[1]) > 19:
        print_usage()
        exit(0)

    game = goboard.GameState.new_game(int(argv[1]))

    human = None
    players = {}
    p = argv[2].lower()
    if p == "hb":
        players = {
            gotypes.Player.black: human,
            gotypes.Player.white: naive.RandomBot()
        }
    elif p == "bh":
        players = {
            gotypes.Player.black: naive.RandomBot(),
            gotypes.Player.white: human
        }
    elif p == "bb":
        players = {
            gotypes.Player.black: naive.RandomBot(),
            gotypes.Player.white: naive.RandomBot()
        }
    else:
        print_usage()
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
                    if match("PA(S)*", human_move):
                        move = goboard.Move.pass_turn()
                    elif match("RE(SIGN)*", human_move):
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
        print_move(player_before, move)
        time.sleep(.3)
    # end of main game loop

    results = compute_game_result(game)
    print("White Final Score: %d" % results.w)
    print("Black Final Score: %.1f" % (results.b - results.komi))
    print(results.winner, "wins!")


if __name__ == '__main__':
    main()