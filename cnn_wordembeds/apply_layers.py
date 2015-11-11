#!/usr/bin/env python

"""
Encode a same-different set using the layers from the given model.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from datetime import datetime
from os import path
import argparse
import cPickle as pickle
import logging
import numpy as np
import sys
import theano
import theano.tensor as T

from data_io import smart_open
import train_cnn
import train_mlp
import train_siamese_cnn
import train_siamese_triplets_cnn

logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("model_dir", type=str, help="model directory")
    parser.add_argument(
        "set", type=str, help="set to perform evaluation on", choices=["train", "dev", "test"]
        )
    parser.add_argument(
        "--batch_size", type=int, help="if not provided, a single batch is used"
        )
    parser.add_argument(
        "--i_layer", type=int, help="the layer of the network to use (default: %(default)s)", default=-1
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def load_model(options_dict):
    if "siamese_triplets" in options_dict["model_dir"]:
        model = train_siamese_triplets_cnn.load_siamese_triplets_cnn(options_dict)
    elif "siamese" in options_dict["model_dir"]:
        model = train_siamese_cnn.load_siamese_cnn(options_dict)
    elif "mlp" in options_dict["model_dir"]:
        model = train_mlp.load_mlp(options_dict)
    else:
        model = train_cnn.load_cnn(options_dict)
    return model


def apply_layers(model_dir, set, batch_size=None, i_layer=-1):

    logger.info(datetime.now())

    # Load the model options
    options_dict_fn = path.join(model_dir, "options_dict.pkl.gz")
    logger.info("Reading: " + options_dict_fn)
    f = smart_open(options_dict_fn)
    options_dict = pickle.load(f)
    # print options_dict
    f.close()

    # Load the dataset
    npz_fn = path.join(options_dict["data_dir"], "swbd." + set + ".npz")
    logger.info("Reading: " + npz_fn)
    npz = np.load(npz_fn)
    logger.info("Loaded " + str(len(npz.keys())) + " segments")

    # Load the model
    if batch_size is not None:
        options_dict["batch_size"] = batch_size
    else:
        options_dict["batch_size"] = len(npz.keys())
    model = load_model(options_dict)

    # Load data into Theano shared variable
    utt_ids = sorted(npz.keys())
    mats = np.array([npz[i] for i in utt_ids])
    logger.info("Data shape: " + str(mats.shape))
    logger.info("Formatting into Theano shared variable")
    shared_x = theano.shared(np.asarray(mats, dtype=theano.config.floatX), borrow=True)

    # Flatten data
    d_in = 39*200
    shared_x = shared_x.reshape((-1, d_in))

    # Compile function for passing segments through CNN layers
    x = model.input  # input to the tied layers
    i_batch = T.lscalar()
    layers_output = model.layers[i_layer].output
    apply_model = theano.function(
        inputs=[i_batch],
        outputs=layers_output,
        givens={
            x: shared_x[
                i_batch * options_dict["batch_size"] : 
                (i_batch + 1) * options_dict["batch_size"]
                ]
            }
        )

    logger.info(datetime.now())

    n_batches = mats.shape[0]/options_dict["batch_size"]
    logger.info("Passing data through in batches: " + str(n_batches))
    layers_outputs = []
    for i_batch in xrange(n_batches):
        batch_layers_outputs = apply_model(i_batch)
        layers_outputs.append(batch_layers_outputs)
    layers_outputs = np.vstack(layers_outputs)
    logger.info("Outputs shape: " + str(layers_outputs.shape))

    layers_output_dict = {}
    # for i , utt_id in enumerate(utt_ids):
    for i in xrange(layers_outputs.shape[0]):
        utt_id = utt_ids[i]
        layers_output_dict[utt_id] = layers_outputs[i]

    logger.info(datetime.now())

    return layers_output_dict


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    logging.basicConfig(level=logging.DEBUG)

    layers_output_dict = apply_layers(args.model_dir, args.set, args.batch_size, args.i_layer)

    layers_output_npz_fn = path.join(
        args.model_dir, "swbd." + args.set + ".layer_" + str(args.i_layer) + ".npz"
        )
    logger.info("Writing: " + layers_output_npz_fn)
    np.savez_compressed(layers_output_npz_fn, **layers_output_dict)


if __name__ == "__main__":
    main()
