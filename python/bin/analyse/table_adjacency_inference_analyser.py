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



def analyse():

    store = []
    with open(os.path.join(gconfig.get_config_param('test_out_path', 'str'), 'inference_output_files.txt')) as f:
        content = f.readlines()
        for i in content:
            with gzip.open(i.strip()) as f:
                data = pickle.load(f)
                for j in data:
                    data_dict = j

                    # TODO: Hassan - you can get everything like this
                    vertex_features  = data_dict['vertex_features']
                    image  = data_dict['image']
                    global_features  = data_dict['global_features']
                    gt_cells_adj_matrix  = data_dict['gt_cells_adj_matrix']
                    gt_rows_adj_matrix  = data_dict['gt_rows_adj_matrix']
                    gt_cols_adj_matrix  = data_dict['gt_cols_adj_matrix']
                    predicted_cells_adj_matrix  = data_dict['predicted_cells_adj_matrix']
                    predicted_rows_adj_matrix  = data_dict['predicted_rows_adj_matrix']
                    predicted_cols_adj_matrix  = data_dict['predicted_cols_adj_matrix']
                    store.append(do_something_on_data())

    # TODO: Hassan
    # Iterate and calculate accuracy etc
    for i in store:
        pass


if __name__ != '__main__':
    print("Can't import  this as a module")
