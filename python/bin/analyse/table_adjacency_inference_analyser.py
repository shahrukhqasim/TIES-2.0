import argparse

def str2bool(v):
    if type(v) == bool:
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Plot clustering model output')
parser.add_argument('input', help="Path to the config file which was used to train")
parser.add_argument('config', help="Config section within the config file")
parser.add_argument('--figures', help="Whether to show 3d plots", default=False)
args = parser.parse_args()

if __name__ != "__main__":
    print("Can't import this file")
    exit(0)

show_3d_figures = str2bool(args.figures)

import matplotlib as mpl
if not show_3d_figures:
    mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import sys
import math
import gzip
import pickle
import configparser as cp
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.backends.backend_pdf
from libs.configuration_manager import ConfigurationManager as gconfig
import libs.plots as plots
import subprocess



def analyse(args):
    gconfig.init(args.input, args.config)
    out_path = gconfig.get_config_param('test_out_path', 'str')
    out_path = os.path.join(out_path, 'visualizations')
    subprocess.call("mkdir -p %s" % (out_path), shell=True)

    store = []
    inference_files = os.path.join(gconfig.get_config_param('test_out_path', 'str'), 'inference_output_files.txt')

    sample_number = 0
    with open(inference_files) as f:
        content = f.readlines()
        for i in content:
            with gzip.open(i.strip()) as f:
                data = pickle.load(f)
                for j in data:
                    result = j
                    image = result['image']
                    ground_truths = result['sampled_ground_truths']
                    predictions = result['sampled_predictions']
                    indices = result['sampled_indices']
                    global_features = result['global_features']
                    vertex_features = result['vertex_features']

                    print(indices[0].shape)
                    print(indices[0])
                    plots.plot_few(out_path, sample_number, result)
                    sample_number += 1


    # TODO: Hassan
    # Iterate and calculate accuracy etc
    for i in store:
        pass


if __name__ != '__main__':
    print("Can't import  this as a module")

parser = argparse.ArgumentParser(description='Run analysis')
parser.add_argument('input', help="Path to config file")
parser.add_argument('config', help="Config section within the config file")
args = parser.parse_args()

analyse(args)
