import os
import tensorflow as tf

N_BOXES = 8
default_log_dir = os.path.join(os.getcwd(), "log")
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
session = tf.Session(config = config)

