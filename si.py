import numpy as np
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

from sample import *
from unet import *
from forward_process import *
from dataset import *

from si4dnn.si4dnn import si
from si4dnn.si4dnn.layers import truncated_interval
from sicore import NaiveInferenceNorm, SelectiveInferenceNorm

def make_mean_filter_kernel(kernel_size: int) -> np.ndarray:
    """Make mean filter kernel for one channel image
    Parameters
    ----------
    kernel_size : int
        width and height of kernel

    Returns
    ----------
    kernel : np.ndarray
        mean filter kernel
    """
    kernel = tf.ones([kernel_size, kernel_size, 1, 1]) / (kernel_size * kernel_size)
    kernel = tf.cast(kernel, dtype=tf.float64)

    return kernel

class AEInferenceNorm(si.SI4DNN):
    def __init__(self, model, config, X, thr: float, kernel_size: int, var: float=1.0):
        super().__init__(model, var)
        self.thr = thr
        self.config = config
        self.initial_noise = tf.random.normal(X.shape, dtype=tf.float64)
        seq = range(0, self.config.model.test_trajectory_steps, self.config.model.skip)
        self.noise_each_seq = [tf.random.normal(X.shape, dtype=tf.float64) for i in range(len(seq))]
        self.input_names = model.input_names
        self.model = model
            
        self.kernel = make_mean_filter_kernel(kernel_size)
            

    def naive_inference(self, X):
        self.construct_hypothesis(X)
        z = tf.tensordot(self.eta, tf.cast(tf.reshape(X[0], [-1]), tf.float64), axes=1) / tf.sqrt(self.si_calculator.eta_sigma_eta)
        p_value = 2 * stats.norm.cdf(-np.abs(z))
        return p_value, z

    def construct_hypothesis(self, X):
        self.shape = X[0].shape
        self.input = X[0]
        with tf. GradientTape(persistent=True) as tape:
            test_trajectory_steps = tf.constant(
                [self.config.model.test_trajectory_steps], dtype=tf.int64
            )
            at = compute_alpha(test_trajectory_steps, self.config)
            
            seq = range(0, self.config.model.test_trajectory_steps, self.config.model.skip)
            seq_next = [-1] + list(seq[:-1])
            
            y0 = X[0]
            y0 = tf.cast(y0, tf.float64)
            
            noisy_image = tf.math.sqrt(at) * y0 + tf.math.sqrt(1 - at) * self.initial_noise
            n = noisy_image.shape[0]
            xs = [noisy_image]

            for index, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
                t = tf.constant(np.ones(n) * i, dtype=tf.int64)
                next_t = tf.constant(np.ones(n) * j, dtype=tf.int64)

                at = compute_alpha(t, self.config)
                at = tf.cast(at, tf.float64)
                at_next = compute_alpha(next_t, self.config)
                at_next = tf.cast(at_next, tf.float64)
                
                t = tf.reshape(t, [1, 1])
                t = tf.cast(t, tf.float64)

                inputs = {self.input_names[0]: xs[-1], self.input_names[1]: t}

                self.output, _ = self.si_model.forward(inputs)
                
                xt = xs[-1]

                self.output = tf.cast(self.output, tf.float64)
                
                et_hat = self.output - tf.math.sqrt(1 - at)
                
                
                x0_t = (xt - et_hat * tf.math.sqrt(1 - at)) / (tf.math.sqrt(at))
                c1 = self.config.model.eta * tf.math.sqrt((1 - at / at_next) * (1 - at_next) / (1 - at))
                c2 = tf.math.sqrt((1 - at_next) - c1 ** 2)
                xt_next = tf.math.sqrt(at_next) * x0_t + c1 * self.noise_each_seq[index] + c2 * et_hat
                xs.append(xt_next)
                self.output = xt_next
        self.final_outputs = xs
        
        input_vec = tf.cast(tf.reshape(X[0], [-1]), tf.float64)
        output_vec = tf.cast(tf.reshape(self.output, [-1]), tf.float64)

        abnormal_error = tf.reshape(input_vec, [*X[0].shape]) - tf.reshape(output_vec, [*X[0].shape])
        
        abnormal_error = tf.nn.depthwise_conv2d(
            abnormal_error, self.kernel, [1, 1, 1, 1], "SAME"
        )
        abnormal_error = tf.reshape(abnormal_error, [-1])
        
        abnormal_index = tf.where(tf.abs(abnormal_error) >= self.thr, 1.0, 0.0)
        abnormal_index = tf.cast(abnormal_index, tf.float64)

        if tf.reduce_sum(abnormal_index) == 0:
            raise si.NoHypothesisError
        
        self.eta = (abnormal_index / tf.reduce_sum(abnormal_index)) - (1.0 - abnormal_index) / (tf.reduce_sum(1.0 - abnormal_index))
        self.eta = tf.cast(self.eta, tf.float64)
        
        abnormal_index = tf.cast(abnormal_index, tf.bool)
        
        self.abnormal_index = abnormal_index

        self.input_vec = input_vec
        self.output_vec = output_vec

        self.si_calculator = SelectiveInferenceNorm(
            self.input_vec, self.var, self.eta, use_tf=True
        )

        self.si_calculator_naive = NaiveInferenceNorm(
            self.input_vec, self.var, self.eta, use_tf=True
        )

        # set upper and lower bound of search range
        sd: float = np.sqrt(self.si_calculator.eta_sigma_eta)
        self.max_tail = sd * 10 + tf.abs(self.si_calculator.stat)

    def model_selector(self, abnormal_index):
        return tf.reduce_all(tf.math.equal(self.abnormal_index, abnormal_index))

    def algorithm(self, a, b, z):
        x = a + b * z
        B, H, W, C = self.shape

        input_x = tf.reshape(tf.constant(x, dtype=tf.float64), [B, H, W, C])
        input_bias = tf.zeros([B, H, W, C], dtype=tf.float64)
        input_a = tf.reshape(tf.constant(a, dtype=tf.float64), [B, H, W, C])
        input_b = tf.reshape(tf.constant(b, dtype=tf.float64), [B, H, W, C])

        input_t = tf.reshape(tf.constant(z, dtype=tf.float64), [B, 1])
        input_t_bias = tf.zeros([B, 1], dtype=tf.float64)
        input_t_a = tf.zeros([B, 1], dtype=tf.float64)
        input_t_b = tf.zeros([B, 1], dtype=tf.float64)

        l = -self.max_tail
        u = self.max_tail

        test_trajectory_steps = tf.constant([self.config.model.test_trajectory_steps], dtype=tf.int64)
        at = compute_alpha(test_trajectory_steps, self.config)
        
        seq = range(0, self.config.model.test_trajectory_steps, self.config.model.skip)
        y0 = input_x
        y0 = tf.cast(y0, tf.float64)
        
        noisy_image = tf.math.sqrt(at) * y0 + tf.math.sqrt(1 - at) * self.initial_noise
        noisy_bias = tf.math.sqrt(at) * input_bias + tf.math.sqrt(1 - at) * self.initial_noise
        noisy_a = tf.math.sqrt(at) * input_a
        noisy_b = tf.math.sqrt(at) * input_b
    
        n = noisy_image.shape[0]
        seq_next = [-1] + list(seq[:-1])
        xs = [noisy_image]
        output_bias, output_a, output_b = noisy_bias, noisy_a, noisy_b

        for index, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            t = tf.constant(np.ones(n) * i, dtype=tf.int64)
            next_t = tf.constant(np.ones(n) * j, dtype=tf.int64)
            
            at = compute_alpha(t, self.config)
            at = tf.cast(at, tf.float64)
            at_next = compute_alpha(next_t, self.config)
            at_next = tf.cast(at_next, tf.float64)
            
            t = tf.reshape(t, [B, 1])
            t = tf.cast(t, tf.float64)
            input_t_bias = t

            xt = xs[-1]
            xt_x = xt
            xt_bias = output_bias
            xt_a = output_a
            xt_b = output_b
            
            input_si = {
                self.input_names[0]: (
                    xt_x,
                    xt_bias,
                    xt_a,
                    xt_b,
                    l,
                    u
                ),
                self.input_names[1]: (
                    t,
                    input_t_bias,
                    input_t_a,
                    input_t_b,
                    l,
                    u
                )
            }
            
            _output_x, _ = self.si_model.forward({self.input_names[0]: xs[-1], self.input_names[1]: t})
            l, u, output, output_si_dict = self.si_model.forward_si(input_si)
            
            assert np.allclose(_output_x[0], output[0])

            output_x, output_bias, output_a, output_b, l, u = output_si_dict[
                self.si_model.output
            ]

            output_x = output_x[0]
            output_bias = output_bias[0]
            output_a = output_a[0]
            output_b = output_b[0]
            l = l[0]
            u = u[0]
            
            yt = tf.math.sqrt(at) * y0 + tf.math.sqrt(1 - at) * output_x
            yt_bias = tf.math.sqrt(at) * y0 + tf.math.sqrt(1 - at) * output_bias
            yt_a = tf.math.sqrt(1 - at) * output_a
            yt_b = tf.math.sqrt(1 - at) * output_b

            noise_t_next = self.noise_each_seq[index]

            et_hat = output_x
            ex_hat_bias = output_bias
            ex_hat_a = output_a
            ex_hat_b = output_b

            x0_t = (xt - et_hat * tf.math.sqrt(1 - at)) / (tf.math.sqrt(at))
            x0_t_bias = (xt_bias - ex_hat_bias * tf.math.sqrt(1 - at)) / (tf.math.sqrt(at))
            x0_t_a = (xt_a - ex_hat_a * tf.math.sqrt(1 - at)) / (tf.math.sqrt(at))
            x0_t_b = (xt_b - ex_hat_b * tf.math.sqrt(1 - at)) / (tf.math.sqrt(at))
                        
            c1 = self.config.model.eta * tf.math.sqrt((1 - at / at_next) * (1 - at_next) / (1 - at))
            c2 = tf.math.sqrt((1 - at_next) - c1 ** 2) #
            
            output_x = tf.math.sqrt(at_next) * x0_t + c1 * noise_t_next + c2 * et_hat
            output_bias = tf.math.sqrt(at_next) * x0_t_bias + c1 * noise_t_next + c2 * ex_hat_bias
            output_a = tf.math.sqrt(at_next) * x0_t_a + c2 * ex_hat_a
            output_b = tf.math.sqrt(at_next) * x0_t_b + c2 * ex_hat_b
            
            xs.append(output_x)
            output = output_x
            _output_x = output_x
        
        if l > u:
            print(l, u)
            print(x)
            assert False
        
        error_x = input_x - output_x
        error_bias = 0 - output_bias
        error_a = input_a - output_a
        error_b = input_b - output_b
        
        smoothed_error_x = tf.nn.depthwise_conv2d(
            error_x, self.kernel, [1, 1, 1, 1], "SAME"
        )
        smoothed_error_bias = tf.nn.depthwise_conv2d(
            error_bias, self.kernel, [1, 1, 1, 1], "SAME"
        )
        smoothed_error_a = tf.nn.depthwise_conv2d(
            error_a, self.kernel, [1, 1, 1, 1], "SAME"
        )
        smoothed_error_b = tf.nn.depthwise_conv2d(
            error_b, self.kernel, [1, 1, 1, 1], "SAME"
        )
        abnormal_index = tf.abs(smoothed_error_x) >= self.thr
        abnormal_index = tf.cast(tf.reshape(abnormal_index, [-1]), tf.bool)

        positive_index = smoothed_error_x >= self.thr

        tTa = tf.where(positive_index, -smoothed_error_a, smoothed_error_a)
        tTb = tf.where(positive_index, -smoothed_error_b, smoothed_error_b)
        event_bias = smoothed_error_bias - self.thr
        event_bias = tf.where(positive_index, -event_bias, event_bias)
        l_positive, u_positive = truncated_interval(tTa, tTb, event_bias)

        assert l_positive < u_positive

        negative_index = smoothed_error_x < -self.thr
        tTa = tf.where(negative_index, smoothed_error_a, -smoothed_error_a)
        tTb = tf.where(negative_index, smoothed_error_b, -smoothed_error_b)
        event_bias = smoothed_error_bias + self.thr
        event_bias = tf.where(negative_index, event_bias, -event_bias)

        l_negative, u_negative = truncated_interval(tTa, tTb, event_bias)
        assert l_negative < u_negative

        l = tf.reduce_max([l_positive, l_negative, l])
        u = tf.reduce_min([u_positive, u_negative, u])

        if l > u:
            print("negative", l_negative, u_negative)
            print("positive", l_positive, u_positive)
            print("normal", l, u)
            assert l < u

        return abnormal_index, (l, u)