# tag::gtp_pachi[]
from dlgo.gtp.play_local import LocalGtpBot
from dlgo.agent.termination import PassWhenOpponentPasses
from dlgo.agent.predict import load_prediction_agent
import h5py
import tensorflow as tf


try:
    with tf.device('GPU:0'):
        bot = load_prediction_agent(h5py.File("./agents/deep_bot.h5", "r"))

        gtp_bot = LocalGtpBot(go_bot=bot, termination=PassWhenOpponentPasses(),
                              handicap=0, opponent='gnugo')
        gtp_bot.run()
except RuntimeError as e:
    print(e)
# end::gtp_pachi[]
