# -*- coding: utf-8 -*-
import dlgo.agent.naive as naive
from dlgo import goboard_slow
from dlgo import gotypes
from dlgo.utils import print_board, print_move
import time
from os import system, name


def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')


def main():
    board_size = 9
    game = goboard_slow.GameState.new_game(board_size)
    bots = {
            gotypes.Player.black: naive.RandomBot(),
            gotypes.Player.white: naive.RandomBot(),
    }
    while not game.is_over():
        time.sleep(.3)
        
        clear()
        print_board(game.board)
        bot_move = bots[game.next_player].select_move(game)
        print_move(game.next_player, bot_move)
        game = game.apply_move(bot_move)
        
    
if __name__ == '__main__':
    main()