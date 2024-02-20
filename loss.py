import tensorflow as tf
import numpy as np
from forward_process import *


def get_loss(e_pred, e):
    return tf.reduce_mean(tf.math.abs(e - e_pred))