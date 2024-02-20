import os
import argparse
import numpy as np
import tensorflow as tf
from scipy.stats import norm
from scipy.stats import kstest
from scipy.optimize import brentq
from scipy import stats
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle
import traceback


from tqdm import tqdm
from omegaconf import OmegaConf
from concurrent.futures import ProcessPoolExecutor

import sicore
from seed import set_seed
from unet import *
from dataset import *
from si import AEInferenceNorm


def compute_threshold(alpha, base ,power):
    def target_func(th):
        log_bonf_alpha = np.log(alpha) - np.log(base) * power
        return np.log(2.0) + norm.logcdf(-th) - log_bonf_alpha
    th = brentq(target_func, 0.0, 1000.0)
    return th


def generate_images_iid(img_size, sigma, signal, num):
    images = []
    for _ in range(num):
        image = rng.normal(0, sigma, (img_size, img_size))
        abnormal_size = int(img_size/3)
        abnormal_x = rng.integers(0, img_size - abnormal_size)
        abnormal_y = rng.integers(0, img_size - abnormal_size)
        image[abnormal_x:abnormal_x+abnormal_size, abnormal_y:abnormal_y+abnormal_size] += signal
        images.append(image)
    images = np.array(images)
    images = images.reshape(num, img_size, img_size, 1)
    return images


def generate_images_corr(img_size, num, signal):
    cov = [[] for _ in range(img_size * img_size)]
    for i in range(img_size*img_size):
        for j in range(img_size*img_size):
            cov[i].append(np.abs(i - j))
    cov = np.array(cov)
    cov = np.power(0.5, cov)

    images = rng.multivariate_normal(
        np.zeros(img_size * img_size), cov, num
    )
    images = images.reshape(num, img_size, img_size, 1)
    for _ in range(num):
        abnormal_size = int(img_size/3)
        abnormal_x = rng.integers(0, img_size - abnormal_size)
        abnormal_y = rng.integers(0, img_size - abnormal_size)
        images[_, abnormal_x:abnormal_x+abnormal_size, abnormal_y:abnormal_y+abnormal_size] += signal
    return images, cov


def fn(args):
    config, (X, category, image_size, thr, var) = args
    
    shape  = (image_size, image_size, 1)
    unet = UNet(
        image_size,
        config=config
    )

    unet((tf.zeros((1, image_size, image_size, 1)), tf.zeros((1, ))))
    unet.load_weights(
        os.path.join(os.getcwd(),
        'checkpoints/',
        f'{category}_{image_size}',
        str(config.model.epochs) + '.h5')
    )                

    ddad = unet.model
    X_fuga = tf.random.normal((1, *shape), dtype=tf.float64)
    T = tf.random.normal((1, 1), dtype=tf.float64)

    if category == 'brain':
        kernel_size = 2
    else:
        kernel_sizes = {8: 2, 16: 2, 32: 2, 64: 2}
        kernel_size = kernel_sizes.get(image_size, 2)

    si_ddad = AEInferenceNorm(ddad, config, X=X_fuga, thr=thr, kernel_size=kernel_size, var=var)
    
    try:
        pp_p_value = si_ddad.inference([X, T], over_conditioning=False).p_value
        oc_p_value = si_ddad.inference([X, T], over_conditioning=True).p_value
        naive_p_value, z = si_ddad.naive_inference([X, T])
        output = si_ddad.final_outputs
        abnormal_index = si_ddad.abnormal_index
        
        corr_z_list = []
        if category == 'brain':
            permutation_p_value = None
        else:
            B = 1000
            cnt = 0
            while cnt < B:
                try:
                    X_shuffled = tf.random.shuffle(tf.reshape(X, [-1]))
                    X_shuffled = tf.reshape(X_shuffled, (1, *shape))
                    _, permutation_z = si_ddad.naive_inference([X_shuffled, T])
                    corr_z_list.append(tf.abs(permutation_z))
                    cnt+=1
                except:
                    continue
            permutation_p_value = 1/B * np.sum(np.array(corr_z_list) > tf.abs(z))
    except:
        print(None)
        traceback.print_exc()
        return None

    return pp_p_value, oc_p_value, naive_p_value, z, output, permutation_p_value, abnormal_index


def experiment(
        config,
        category : str,
        image_size : int,
        thr,
        signal,
        alpha,
        seed,
        number_of_workers,
        number_of_iter,
        **kwargs
        ):

    if category == 'iid':
        input_image_list = generate_images_iid(image_size, sigma=1, signal=signal, num=number_of_iter)
        var = 1.0
    elif category == 'corr':
        input_image_list, cov = generate_images_corr(image_size, number_of_iter, signal)
        var = cov    

    with ProcessPoolExecutor(max_workers=number_of_workers) as executor:
        args = ((config,
                (input_image_list[i:i+1,:,:,:],
                 category,
                 image_size,
                 thr,
                 var)) for i in range(number_of_iter))
        outputs = list(tqdm(executor.map(fn, args), total=number_of_iter))

        p_values = []
        input_images = []
        output_images = []
        permutation_p_values = []
        abnormal_index = []
        for i in range(number_of_iter):
            if outputs[i] is None or None in outputs[i][0:4] or outputs[i][6] is None:
                continue
            p_values.append(outputs[i][0:4])
            input_images.append(input_image_list[i:i+1, :, :, :])
            output_images.append(outputs[i][4])
            permutation_p_values.append(outputs[i][5])
            abnormal_index.append(outputs[i][6])
            
        result = np.array([p_value for p_value in p_values if p_value is not None])
        pp_p_values = result[:, 0]
        oc_p_values = result[:, 1]
        naive_p_values = result[:, 2]
        z = result[:, 3]
        
    ks_pp_p_value = kstest(pp_p_values, "uniform")[1]
    ks_oc_p_value = kstest(oc_p_values, "uniform")[1]
    ks_naive_p_value = kstest(naive_p_values, "uniform")[1]

    result_dict = {
        "category":category,
        "image_size":image_size,
        "signal":signal,
        "thr":thr,
        "alpha":alpha,
        "number_of_iter":number_of_iter,
        "seed": seed,
        "pp_p_values": pp_p_values,
        "oc_p_values" : oc_p_values,
        "naive_p_values" : naive_p_values,
        "z" : z,
        "ks_p": ks_pp_p_value,
        "ks_oc_p": ks_oc_p_value,
        "ks_naive_p": ks_naive_p_value,
        "num_of_p_values": len(pp_p_values),
        "abnormal_index": abnormal_index,
    }

    result_dict['input_images'] = input_images
    result_dict['output_images'] = output_images

    result_dict['permutation_p_values'] = permutation_p_values    


    if category == 'brain':
        print(f'./results/result_brain'+f'_{image_size}'+f'_seed{seed}'+'.pickle')
        with open(f'./results/result_brain'
                  + f'_{image_size}'
                  + f'_seed{seed}' + '.pickle', 'wb') as f:
            pickle.dump(result_dict, f)
    elif category == 'iid':
        print(f'./results/result_iid'+f'_{image_size}'+f'_seed{seed}'+'.pickle')
        with open(f'./results/result_iid' 
                  + f'_{image_size}' 
                  + f'_signal{int(signal)}' 
                  + f'_seed{seed}'+'.pickle', 'wb') as f:
            pickle.dump(result_dict, f)
    elif category == 'corr':
        print(f'./results/result_corr'+f'_{image_size}'+f'_seed{seed}'+'.pickle')
        with open(f'./results/result_corr'
                  + f'_{image_size}'
                  + f'_signal{int(signal)}'
                  + f'_seed{seed}' + '.pickle', 'wb') as f:
            pickle.dump(result_dict, f)


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser('DDAD')    
    cmdline_parser.add_argument(
            '-cfg', '--config', 
            default= os.path.join(os.path.dirname(os.path.abspath(__file__)),'config.yaml'), 
            help='config file'
        )
    cmdline_parser.add_argument('-size', '--size', type=int)
    cmdline_parser.add_argument('-thr', '--thr', type=float)
    cmdline_parser.add_argument('-signal', '--signal', type=float)
    cmdline_parser.add_argument('-alpha', '--alpha', type=float)
    cmdline_parser.add_argument('-workers', '--workers', type=int)
    cmdline_parser.add_argument('-iter', '--iter', type=int)
    cmdline_parser.add_argument('-seed', '--seed', type=int)

    args, unknowns = cmdline_parser.parse_known_args()
    config = OmegaConf.load(args.config)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    
    experiment(
        config,
        category = config.data.category,
        image_size = args.size,
        thr = args.thr,
        signal = args.signal,
        alpha = args.alpha,
        seed = args.seed,
        number_of_workers=args.workers,
        number_of_iter = args.iter,
    )
