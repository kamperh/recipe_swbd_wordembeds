#!/usr/bin/env python

"""
Summarize the results given a directory containing the models from a sweep.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from os import path
import argparse
import cPickle as pickle
import glob
import sys

from data_io import smart_open

data_dir_filter = None  # if given, `data_dir` should contain this string
options_monitor = ["batch_size", "n_max_epochs", "rnd_seed", "margin"]


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument(
        "model_basedir", type=str,
        help="the base directory of the models; multiple directories can be given separated by commas"
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

    if "," in args.model_basedir:
        directory_list = []
        for model_basedir in args.model_basedir.split(","):
            directory_list += glob.glob(path.join(model_basedir, "*"))
        print directory_list
    else:
        directory_list = glob.glob(path.join(args.model_basedir, "*"))

    # Get results from directories
    results = []  # list of (dir, option_value_dict, performance)
    for d in directory_list:
        if path.isdir(d):
            hash = path.split(d)[-1]
            # print d, hash

            options_dict_fn = path.join(d, "options_dict.pkl.gz")
            if not path.isfile(options_dict_fn):
                continue
            print "Reading:", options_dict_fn
            f = smart_open(options_dict_fn)
            options_dict = pickle.load(f)
            f.close()

            # Data  filter
            if data_dir_filter is not None:
                if not data_dir_filter in options_dict["data_dir"]:
                    continue

            # Read average precision
            ap_fn = path.join(d, "dev_ap.txt")
            if not path.isfile(ap_fn):
                continue
            with open(ap_fn) as f:
                ap = float(f.readline().strip())

            # Get the options we are interested in
            options = {}
            if "min_count" in options_dict:
                options["min_count"] = options_dict["min_count"]
            else:
                options["min_count"] = None
            if "conv_layer_specs" in options_dict:
                options["n_cnn_units"] = options_dict["conv_layer_specs"][0]["filter_shape"][0]
            else:
                options["n_cnn_units"] = None
            options["n_hidden_units"] = options_dict["hidden_layer_specs"][0]["units"]
            options["n_hidden_layers"] = len(options_dict["hidden_layer_specs"])
            options["n_hidden_units_final_layer"] = options_dict["hidden_layer_specs"][-1]["units"]
            for key in options_monitor:
                if key in options_dict:
                    options[key] = options_dict[key]
                else:
                    options[key] = None

            results.append((d, options, ap))

    # Try to sort the results according to the option_value_dict
    results = sorted(results, key=lambda i:i[1].values())

    # Present results
    options = results[0][1].keys()
    print "Possible options:", options
    print_options = sorted(options)  # or can give a filtered list here
    print print_options
    print
    print "-"*39
    print "# Directory\t" + "\t".join(print_options)  + "\tDev AP"
    for dir, options, ap in results:
        print dir + "\t" + "\t".join([str(options[i]) for i in print_options]) + "\t" + str(ap)
    print "-"*39


if __name__ == "__main__":
    main()
