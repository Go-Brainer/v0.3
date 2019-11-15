from keras.layers import Activation, BatchNormalization
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.models import Model
from dlgo import scoring
from dlgo import zero
from dlgo import kerasutil
from dlgo.goboard_fast import GameState, Player, Point
from dlgo.utils import print_board
from os import name, system

import h5py
import tensorflow as tf

def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')

# tag::zero_simulate[]
def simulate_game(
        board_size,
        black_agent, black_collector,
        white_agent, white_collector):
    print('Starting the game!')
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_agent,
        Player.white: white_agent,
    }

    black_collector.begin_episode()
    white_collector.begin_episode()
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)


    game_result = scoring.compute_game_result(game)
    print_board(game.board)
    game.print_game_results()

    if game_result.winner == Player.black:
        black_collector.complete_episode(1)
        white_collector.complete_episode(-1)
    else:
        black_collector.complete_episode(-1)
        white_collector.complete_episode(1)
# end::zero_simulate[]


# tag::zero_model[]


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_soft_device_placement(True)
    except RuntimeError as e:
        print(e)

board_size = 9
encoder = zero.ZeroEncoder(board_size)
board_input = Input(shape=encoder.shape(), name='board_input')
pb = board_input
for i in range(4):                     # <1>
    pb = Conv2D(64, (3, 3),            # <1>
        padding='same',                # <1>
        data_format='channels_first',  # <1>
        activation='relu')(pb)         # <1>

policy_conv = Conv2D(2, (1, 1), data_format='channels_first', activation='relu')(pb)
policy_flat = Flatten()(policy_conv)
policy_output = Dense(encoder.num_moves(), activation='softmax')( policy_flat)
value_conv = Conv2D(1, (1, 1), data_format='channels_first', activation='relu')(pb)                           # <3>
value_flat = Flatten()(value_conv)                       # <3>
value_hidden = Dense(256, activation='relu')(value_flat) # <3>
value_output = Dense(1, activation='tanh')(value_hidden) # <3>

model = Model(
    inputs=[board_input],
    outputs=[policy_output, value_output])
#end::zero_model[]

# model_file = h5py.File("./zero_bot_0.h5", "r")
# model = zero.load_zero_agent(model_file)


#tag::zero_train[]
black_agent = zero.ZeroAgent(
    model, encoder, rounds_per_move=5, c=10)  # <4>
white_agent = zero.ZeroAgent(
    model, encoder, rounds_per_move=5, c=10)

c1 = zero.ZeroExperienceCollector()
c2 = zero.ZeroExperienceCollector()
black_agent.set_collector(c1)
white_agent.set_collector(c2)

num_games = 100
for i in range(num_games):   # <5>
    print("Simulating game %d/%d\n" % (i+1, num_games))
    simulate_game(board_size, black_agent, c1, white_agent, c2)

exp = zero.combine_experience([c1, c2])
black_agent.train(exp, 0.01, 512)
black_agent.serialize(h5py.File("./zero_bot_9.h5", "w"))
# end::zero_train[]
