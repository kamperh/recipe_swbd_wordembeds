#!/usr/bin/env python

"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from datetime import datetime
from os import path
import argparse
import os
import sys

import sweep_options
import train_siamese_triplets_cnn


#-----------------------------------------------------------------------------#
#                           DEFAULT TRAINING OPTIONS                          #
#-----------------------------------------------------------------------------#

default_options_dict = {
    "data_dir": "data/icassp15.0",
    # "data_dir": "data/tmp",
    "log_to_file": True,
    "n_same_pairs": int(100e3), # if None, all same pairs are used
    "rnd_seed": 42,
    "batch_size": 1024,
    "n_max_epochs": 20,
    "l1_weight": 0.0,
    "l2_weight": 0.0,
    "learning_rule": {
        "type": "adadelta",     # can be "momentum", "adadgrad", "adadelta"
        "rho": 0.9,             # parameters specific to learning rule
        "epsilon": 1e-6
        },
    "dropout_rates": None,      # a list of rates for each layer or None
    "conv_layer_specs": [       # activation can be "sigmoid", "tanh", "relu", "linear"
        # {"filter_shape": (96, 1, 39, 9), "pool_shape": (1, 4), "activation": "relu"},
        # {"filter_shape": (96, 96, 1, 9), "pool_shape": (1, 4), "activation": "relu"},
        {"filter_shape": (96, 1, 39, 9), "pool_shape": (1, 3), "activation": "relu"},
        {"filter_shape": (96, 96, 1, 8), "pool_shape": (1, 3), "activation": "relu"},
        ],
    "hidden_layer_specs": [
        {"units": 2048, "activation": "relu"},
        {"units": 2048, "activation": "linear"}
        ],
    "loss": "hinge_cos",        # can be "hinge_cos"
    "margin": 0.15,              # margin for hinge loss
    }


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument(
        "model_basedir", type=str,
        help="the different models are written to this directory"
        )
    parser.add_argument(
        "--mode", help="mode to use for spawning jobs (default: %(default)s)",
        choices=["serial", "parallel"], default="serial"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    default_options_dict["model_dir"] = args.model_basedir

    # Sweep options
    n_hidden_units = [2048]  # if None, this is not swept
    n_hidden_layers = None  # [3]
    n_cnn_units = None
    n_hidden_units_final_layer = [15]  # [10, 20, 50, 100, 150, 200, 512, 1024, 2048, 4096]
    sweep_options_dict = {
        # "margin": [0.15],
        # "batch_size": [1024],
        "rnd_seed": [45]# range(41, 46)
        }

    print datetime.now()

    sweep_options.sweep_nn_options(
        default_options_dict,
        train_siamese_triplets_cnn.train_siamese_triplets_cnn,
        sweep_options_dict=sweep_options_dict,
        n_cnn_units=n_cnn_units,
        n_hidden_units=n_hidden_units,
        n_hidden_layers=n_hidden_layers,
        n_hidden_units_final_layer=n_hidden_units_final_layer,
        mode=args.mode
        )

    print datetime.now()


if __name__ == "__main__":
    main()
