#!/usr/bin/env python

"""
Analyze the CNN layer(s) of a given model.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from os import path
import argparse
import cPickle as pickle
import matplotlib.pyplot as plt
import PIL.Image as Image
import sys

sys.path.append(path.join("..", "couscous"))

from couscous import plotting
from data_io import smart_open
import train_mlp


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("model_dir", type=str, help="model directory")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    model_fn = path.join(args.model_dir, "model.pkl.gz")
    options_dict_fn = path.join(args.model_dir, "options_dict.pkl.gz")
    record_dict_fn = path.join(args.model_dir, "record_dict.pkl.gz")

    print "Reading:", options_dict_fn
    f = smart_open(options_dict_fn)
    options_dict = pickle.load(f)
    f.close()

    print "Reading:", record_dict_fn
    f = smart_open(record_dict_fn)
    record_dict = pickle.load(f)
    f.close()

    plotting.plot_record_dict(record_dict)

    model = train_mlp.load_mlp(options_dict)

    # Plot some filters
    analyze_layer = 0
    W =  model.layers[analyze_layer].W.get_value(borrow=True).T
    plot_fn = path.join(args.model_dir, "filters.layer_" + str(analyze_layer) + ".png")
    image = Image.fromarray(plotting.tile_images(
        W,
        image_shape=(39, 200), tile_shape=(5, 6)
        ))
    print("Saving: " + plot_fn)
    image.save(plot_fn)
    plt.figure()
    plt.imshow(image, cmap=plt.cm.Greys_r, interpolation="nearest")

    analyze_layer = -1
    W = model.layers[analyze_layer].W.get_value(borrow=True)
    plot_fn = path.join(args.model_dir, "filters.layer_" + str(analyze_layer) + ".png")
    image = Image.fromarray(plotting.array_to_pixels(W))
    image.save(plot_fn)
    print("Saving: " + plot_fn)
    plt.figure()
    plt.imshow(image, cmap=plt.cm.Greys_r, interpolation="nearest")
    # plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
