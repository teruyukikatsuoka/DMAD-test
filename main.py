import numpy as np
import os
import argparse
from unet import *
from train import trainer
from omegaconf import OmegaConf

import sicore
from seed import set_seed

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def build_model(config):
    unet = UNet(
        config.data.image_size,
        config=config
    )
    return unet

def train(args):
    config = OmegaConf.load(args.config)
    unet = build_model(config)
    trainer(unet, config.data.category, config)

def parse_args():
    cmdline_parser = argparse.ArgumentParser('DMAD')    
    cmdline_parser.add_argument('-cfg', '--config', 
                                default= os.path.join(os.path.dirname(os.path.abspath(__file__)),'config.yaml'), 
                                help='config file')
    cmdline_parser.add_argument('--train', 
                                default= False, 
                                help='Train the diffusion model')
    args, unknowns = cmdline_parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    config = OmegaConf.load(args.config)
    set_seed(config.data.seed)
    rng = np.random.default_rng(seed=config.data.seed)

    if args.train:
        print('Training...')
        train(args)
