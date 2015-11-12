#!/usr/bin/env python

"""
Train an MLP for word classification.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from datetime import datetime
from os import path
from scipy.spatial.distance import pdist
from theano.tensor.shared_randomstreams import RandomStreams
import argparse
import cPickle as pickle
import logging
import numpy as np
import os
import sys
import theano
import theano.tensor as T

sys.path.append(path.join("..", "..", "src", "couscous"))

from couscous import logistic, mlp, theano_utils, training
import apply_layers
import data_io
import samediff

logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------#
#                           DEFAULT TRAINING OPTIONS                          #
#-----------------------------------------------------------------------------#

default_options_dict = {
    "data_dir": "data/icassp15.0",
    # "data_dir": "data/tmp",
    "min_count": 3,             # minimum number of times a training label needs to occur
    "rnd_seed": 42,
    "batch_size": 30,
    "n_max_epochs": 50,
    "l1_weight": 0.0,
    "l2_weight": 0.0,
    "learning_rule": {
        "type": "adadelta",     # can be "momentum", "adadgrad", "adadelta"
        "rho": 0.9,             # parameters specific to learning rule
        "epsilon": 1e-6         
        },
    # "learning_rule": {
    #     "type": "momentum",
    #     "learning_rate": 0.01,
    #     "momentum": 0.9
    #     },
    "dropout_rates": None,      # a list of rates for each layer or None
    "hidden_layer_specs": [     # activation can be "sigmoid", "tanh", "relu", "linear"
        {"units": 2048, "activation": "relu"},
        {"units": 2048, "activation": "relu"},
        ],
    "i_layer_eval": -1,         # -1 corresponds to the softmax layer, -2 to the last hidden layer
    }


#-----------------------------------------------------------------------------#
#                        TRAINING FUNCTIONS AND CLASSES                       #
#-----------------------------------------------------------------------------#

def train_mlp(options_dict):
    """Train and save a word classifier MLP."""

    # Preliminary

    logger.info(datetime.now())

    if not path.isdir(options_dict["model_dir"]):
        os.makedirs(options_dict["model_dir"])

    if "log_to_file" in options_dict and options_dict["log_to_file"] is True:
        log_fn = path.join(options_dict["model_dir"], "log")
        print "Writing:", log_fn
        root_logger = logging.getLogger()
        if len(root_logger.handlers) > 0:
            root_logger.removeHandler(root_logger.handlers[0])  # close open file handler
        logging.basicConfig(filename=log_fn, level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.DEBUG)

    rng = np.random.RandomState(options_dict["rnd_seed"])
    if options_dict["dropout_rates"] is not None:
        srng = RandomStreams(seed=options_dict["rnd_seed"])
    else:
        srng = None


    # Load and format data

    # Load into shared variables
    datasets, word_to_i_map = data_io.load_swbd_labelled(rng, options_dict["data_dir"], options_dict["min_count"])
    train_x, train_y = datasets[0]
    dev_x, dev_y = datasets[1]
    test_x, test_y = datasets[2]

    # Get batch sizes and iterators
    class BatchIterator(object):
        def __init__(self, n_batches):
            self.n_batches = n_batches
        def __iter__(self):
            for i_batch in xrange(self.n_batches):
                yield [i_batch]
    n_train_batches = train_x.get_value(borrow=True).shape[0] / options_dict["batch_size"]
    n_dev_batches = dev_x.get_value(borrow=True).shape[0] / options_dict["batch_size"]
    n_test_batches = test_x.get_value(borrow=True).shape[0] / options_dict["batch_size"]
    train_batch_iterator = BatchIterator(n_train_batches)
    validate_batch_iterator = BatchIterator(n_dev_batches)
    test_batch_iterator = BatchIterator(n_test_batches)

    # Flatten data
    d_in = 39*200
    train_x = train_x.reshape((-1, d_in))
    dev_x = dev_x.reshape((-1, d_in))
    test_x = test_x.reshape((-1, d_in))
    d_out = len(word_to_i_map)
    options_dict["d_out"] = d_out

    # Save `options_dict`
    options_dict_fn = path.join(options_dict["model_dir"], "options_dict.pkl.gz")
    logger.info("Saving options: " + options_dict_fn)
    f = data_io.smart_open(options_dict_fn, "wb")
    pickle.dump(options_dict, f, -1)
    f.close()

    logger.info("Options: " + str(options_dict))


    # Setup model

    logger.info("Building MLP")

    # Symbolic variables
    i_batch = T.lscalar()   # batch index
    x = T.matrix("x")       # flattened data of shape (n_data, d_in)
    y = T.ivector("y")      # labels

    # Build model
    logger.info("No. of word type targets: " + str(options_dict["d_out"]))
    model = mlp.MLP(
        rng, x, d_in, options_dict["d_out"],
        options_dict["hidden_layer_specs"], srng, options_dict["dropout_rates"]
        )
    if options_dict["dropout_rates"] is not None:
        loss = model.dropout_negative_log_likelihood(y)
    else:
        loss = model.negative_log_likelihood(y)
    error = model.errors(y)

    # Add regularization
    if options_dict["l1_weight"] > 0. or options_dict["l2_weight"] > 0.:
        loss = loss + options_dict["l1_weight"]*model.l1 + options_dict["l2_weight"]* model.l2

    # Compile test functions
    outputs = [error, loss]
    validate_model = theano.function(
        inputs=[i_batch],
        outputs=outputs,
        givens={
            x: dev_x[i_batch * options_dict["batch_size"]: (i_batch + 1) * options_dict["batch_size"]],
            y: dev_y[i_batch * options_dict["batch_size"]: (i_batch + 1) * options_dict["batch_size"]]
            }
        )
    test_model = theano.function(
        inputs=[i_batch],
        outputs=outputs,
        givens={
            x: test_x[i_batch * options_dict["batch_size"]: (i_batch + 1) * options_dict["batch_size"]],
            y: test_y[i_batch * options_dict["batch_size"]: (i_batch + 1) * options_dict["batch_size"]]
            }
        )

    # Gradients and training updates
    parameters = model.parameters
    gradients = T.grad(loss, parameters)
    learning_rule = options_dict["learning_rule"]
    if learning_rule["type"] == "adadelta":
        updates = training.learning_rule_adadelta(
            parameters, gradients, learning_rule["rho"], learning_rule["epsilon"]
            )
    elif learning_rule["type"] == "momentum":
        updates = training.learning_rule_momentum(
            parameters, gradients, learning_rule["learning_rate"], learning_rule["momentum"]
            )
    else:
        assert False, "Invalid learning rule: " + learning_rule["type"]

    # Compile training function
    train_model = theano.function(
        inputs=[i_batch],
        outputs=outputs,
        updates=updates,
        givens={
            x: train_x[i_batch * options_dict["batch_size"]: (i_batch + 1) * options_dict["batch_size"]],
            y: train_y[i_batch * options_dict["batch_size"]: (i_batch + 1) * options_dict["batch_size"]]
            },
        )


    # Train model

    logger.info("Training MLP")
    record_dict_fn = path.join(options_dict["model_dir"], "record_dict.pkl.gz")
    record_dict = training.train_fixed_epochs_with_validation(
        options_dict["n_max_epochs"],
        train_model=train_model,
        train_batch_iterator=train_batch_iterator,
        validate_model=validate_model,
        validate_batch_iterator=validate_batch_iterator,
        test_model=test_model,
        test_batch_iterator=test_batch_iterator,
        save_model_func=model.save,
        save_model_fn=path.join(options_dict["model_dir"], "model.pkl.gz"),
        record_dict_fn=record_dict_fn,
        )


    # Extrinsic evaluation

    # Pass data trough model
    logger.info("Performing same-different evaluation")
    layers_output_dict = apply_layers.apply_layers(
        options_dict["model_dir"], "dev", batch_size=645, i_layer=options_dict["i_layer_eval"]
        )
    utt_ids = sorted(layers_output_dict.keys())
    embeddings = np.array([layers_output_dict[i] for i in utt_ids])
    labels = data_io.swbd_utts_to_labels(utt_ids)

    # Perform same-different
    distances = pdist(embeddings, metric="cosine")
    matches = samediff.generate_matches_array(labels)
    ap, prb = samediff.average_precision(distances[matches == True], distances[matches == False])
    logger.info("Validation average precision: " + str(ap))
    ap_fn = path.join(options_dict["model_dir"], "dev_ap.txt")
    with open(ap_fn, "w") as f:
        f.write(str(ap) + "\n")

    # # Pass data through model
    # logger.info("Performing same-different evaluation")
    # npz_fn = path.join(options_dict["data_dir"], "swbd.dev.npz")
    # extr_dev_x, utt_ids = data_io.load_npz(npz_fn)
    # n_extr_dev_batches = extr_dev_x.get_value(borrow=True).shape[0] / options_dict["batch_size"]
    # extr_dev_batch_iterator = BatchIterator(n_extr_dev_batches)
    # extr_dev_x = extr_dev_x.reshape((-1, d_in))
    # extr_validate_model = theano.function(
    #     inputs=[i_batch],
    #     outputs=model.layers[-1].output,
    #     givens={x: extr_dev_x[i_batch * options_dict["batch_size"]: (i_batch + 1) * options_dict["batch_size"]]}
    #     )

    # # Perform same-different
    # def extrinsic_validate():
    #     embeddings = np.vstack([extr_validate_model(*batch) for batch in extr_dev_batch_iterator])
    #     labels = data_io.swbd_utts_to_labels(
    #         utt_ids[:embeddings.shape[0]]
    #         )  # because of the batch size, the last few elements might be missing
    #     distances = pdist(embeddings, metric="cosine")
    #     matches = samediff.generate_matches_array(labels)
    #     ap, prb = samediff.average_precision(distances[matches == True], distances[matches == False])
    #     return ap
    # logger.info("Validation average precision: " + str(extrinsic_validate()))


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("model_dir", type=str, help="directory to write the model to")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def load_mlp(options_dict):
    
    model_fn = path.join(options_dict["model_dir"], "model.pkl.gz")

    # Symbolic variables
    x = T.matrix("x")       # flattened data of shape (n_data, d_in)
    y = T.ivector("y")      # labels

    # Random number generators
    rng = np.random.RandomState(options_dict["rnd_seed"])

    # Build model
    d_in = 39*200
    model = mlp.MLP(
        rng, x, d_in, options_dict["d_out"],
        options_dict["hidden_layer_specs"],
        dropout_rates=None  # dropout is not performed after training
        )

    # Load saved parameters
    logger.info("Reading: " + model_fn)
    f = data_io.smart_open(model_fn)
    model.load(f)
    f.close()

    return model


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # logging.basicConfig(level=logging.DEBUG)

    # Set options
    options_dict = default_options_dict.copy()
    options_dict["model_dir"] = args.model_dir

    # Train and save the model and options
    train_mlp(options_dict)


if __name__ == "__main__":
    main()


