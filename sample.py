# import torch
from forward_process import *
import numpy as np
import tensorflow as tf


def Reconstruction(y0, x, seq, model, config, w):
    tf.random.set_seed(config.data.seed)
    seq = range(0, config.model.test_trajectory_steps, config.model.skip)
    noise_each_seq = [tf.random.normal(y0.shape, dtype=tf.float64) for i in range(len(seq))]

    with tf.GradientTape(persistent=True) as tape:
        n = x.shape[0]
        seq_next = [-1] + list(seq[:-1])
        xs = [x]

        for index, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            t = tf.constant(np.ones(n) * i, dtype=tf.int64)
            next_t = tf.constant(np.ones(n) * j, dtype=tf.int64)
            at = compute_alpha(t, config)
            at_next = compute_alpha(next_t, config)
            xt = xs[-1]
            
            et = model.call((xt, t), training=False)
            
            at = tf.cast(at, tf.float64)
            y0 = tf.cast(y0, tf.float64)
            et = tf.cast(et, tf.float64)
            
            et_hat = et
            x0_t = (xt - et_hat * tf.math.sqrt(1 - at)) / (tf.math.sqrt(at))
            c1 = config.model.eta * tf.math.sqrt((1 - at / at_next) * (1 - at_next) / (1 - at))
            c2 = tf.math.sqrt((1 - at_next) - c1 ** 2)

            xt_next = tf.math.sqrt(at_next) * x0_t + c1 * noise_each_seq[index] + c2 * et_hat
            xs.append(xt_next)
    return xs

def compute_alpha(t, config):
    betas = tf.linspace(config.model.beta_start, config.model.beta_end, config.model.trajectory_steps)
    betas = tf.cast(betas, dtype=tf.float64)
    beta = tf.concat([tf.zeros(1, dtype=tf.float64), betas], axis=0)
    
    a = tf.math.cumprod(1.0 - beta)
    a = tf.gather(a, t+1)
    a = tf.reshape(a, [-1, 1, 1, 1])
    
    return a
