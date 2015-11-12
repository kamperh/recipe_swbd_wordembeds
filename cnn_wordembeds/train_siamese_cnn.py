#!/usr/bin/env python

"""
Train a Siamese CNN on same and different pairs from Switchboard.

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
import scipy.spatial.distance as distance
import sys
import theano
import theano.tensor as T

sys.path.append(path.join("..", "..", "src", "couscous"))

from couscous import siamese, theano_utils, training
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
    "n_same_pairs": int(100e3), # if None, all same pairs are used
    "rnd_seed": 42,
    "batch_size": 2048,
    "n_max_epochs": 20,
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
    "conv_layer_specs": [       # activation can be "sigmoid", "tanh", "relu", "linear"
        {"filter_shape": (96, 1, 39, 9), "pool_shape": (1, 3), "activation": "relu"},
        {"filter_shape": (96, 96, 1, 8), "pool_shape": (1, 3), "activation": "relu"},
        ],
    "hidden_layer_specs": [
        {"units": 2048, "activation": "relu"},
        {"units": 1024, "activation": "linear"}
        ],
    "loss": "cos_cos2",         # can be "cos_cos", "cos_cos2"
    }


#-----------------------------------------------------------------------------#
#                        TRAINING FUNCTIONS AND CLASSES                       #
#-----------------------------------------------------------------------------#

class BatchIteratorSameDifferent(object):
    """Sample a set of different pairs for every batch."""
    
    def __init__(self, rng, matches_vec, batch_size,
            sample_diff_every_epoch=True, n_same_pairs=None):
        """
        If `n_same_pairs` is given, this number of same pairs is sampled,
        otherwise all same pairs are used.
        """
        self.rng = rng
        self.matches_vec = matches_vec
        self.batch_size = batch_size

        if n_same_pairs is None:
            # Use all pairs
            I, J = np.where(np.triu(distance.squareform(matches_vec)))  # indices of same pairs
        else:
            # Sample same pairs
            n_pairs = min(n_same_pairs, len(np.where(matches_vec == True)[0]))
            same_sample = self.rng.choice(
                np.where(matches_vec == True)[0], size=n_pairs, replace=False
                )
            same_vec = np.zeros(self.matches_vec.shape[0], dtype=np.bool)
            same_vec[same_sample] = True
            I, J = np.where(np.triu(distance.squareform(same_vec)))

        self.x1_same_indices = []
        self.x2_same_indices = []
        for i, j in zip(I, J):
            self.x1_same_indices.append(i)
            self.x2_same_indices.append(j)

        if not sample_diff_every_epoch:
            self.x1_diff_indices, self.x2_diff_indices = self._sample_diff_pairs()
        self.sample_diff_every_epoch = sample_diff_every_epoch

    def _sample_diff_pairs(self):
        x1_diff_indices = []
        x2_diff_indices = []
        different_sample = self.rng.choice(
            np.where(self.matches_vec == False)[0],
            size=len(self.x1_same_indices), replace=False
            )
        different_vec = np.zeros(self.matches_vec.shape[0], dtype=np.bool)
        different_vec[different_sample] = True
        I, J = np.where(np.triu(distance.squareform(different_vec)))
        for i, j in zip(I, J):
            x1_diff_indices.append(i)
            x2_diff_indices.append(j)
        return x1_diff_indices, x2_diff_indices

    def __iter__(self):

        # Sample different pairs for this epoch
        if self.sample_diff_every_epoch:
            x1_diff_indices, x2_diff_indices = self._sample_diff_pairs()
        else:
            x1_diff_indices = self.x1_diff_indices
            x2_diff_indices = self.x2_diff_indices
        # x1_diff_indices = []
        # x2_diff_indices = []
        # different_sample = self.rng.choice(
        #     np.where(self.matches_vec == False)[0],
        #     size=len(self.x1_same_indices), replace=False
        #     )
        # different_vec = np.zeros(self.matches_vec.shape[0], dtype=np.bool)
        # different_vec[different_sample] = True
        # I, J = np.where(np.triu(distance.squareform(different_vec)))
        # for i, j in zip(I, J):
        #     x1_diff_indices.append(i)
        #     x2_diff_indices.append(j)

        # Generate the per-batch indices
        n_batches = len(self.x1_same_indices)*2 / self.batch_size
        for i_batch in xrange(n_batches):

            # Pairs for this batch
            batch_x1_indices = np.array(
                self.x1_same_indices[
                    i_batch * self.batch_size / 2: (i_batch + 1) * self.batch_size / 2
                    ] +
                x1_diff_indices[
                    i_batch * self.batch_size / 2: (i_batch + 1) * self.batch_size / 2
                    ],
                dtype=np.int32
                )
            batch_x2_indices = np.array(
                self.x2_same_indices[
                    i_batch * self.batch_size / 2: (i_batch + 1) * self.batch_size / 2
                    ] +
                x2_diff_indices[
                    i_batch * self.batch_size / 2: (i_batch + 1) * self.batch_size / 2
                    ],
                dtype=np.int32
                )
            batch_y = np.array([1]*(self.batch_size/2) + [0]*(self.batch_size/2), dtype=np.int32)

            # assert len(batch_y) == len(batch_x2_indices)
            # # Sample different pairs
            # different_sample = self.rng.choice(
            #     np.where(self.matches_vec == False)[0], size=self.batch_size / 2, replace=False
            #     )
            # different_vec = np.zeros(self.matches_vec.shape[0], dtype=np.bool)
            # different_vec[different_sample] = True
            # I, J = np.where(np.triu(distance.squareform(different_vec)))
            # for i, j in zip(I, J):
            #     batch_x1_indices.append(i)
            #     batch_x2_indices.append(j)
            #     batch_y.append(0)
            # batch_x1_indices = np.array(batch_x1_indices, dtype=np.int32)
            # batch_x2_indices = np.array(batch_x2_indices, dtype=np.int32)
            # batch_y = np.array(batch_y, dtype=np.int32)

            # Now shuffle everything in this batch
            shuffle_order = np.arange(batch_y.shape[0])
            self.rng.shuffle(shuffle_order)
            batch_x1_indices = batch_x1_indices[shuffle_order]
            batch_x2_indices = batch_x2_indices[shuffle_order]
            batch_y = batch_y[shuffle_order]

            yield (batch_x1_indices, batch_x2_indices, batch_y)


def train_siamese_cnn(options_dict):

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

    options_dict_fn = path.join(options_dict["model_dir"], "options_dict.pkl.gz")
    logger.info("Saving options: " + options_dict_fn)
    f = data_io.smart_open(options_dict_fn, "wb")
    pickle.dump(options_dict, f, -1)
    f.close()

    logger.info("Options: " + str(options_dict))


    # Load and format data

    # Load into shared variables
    datasets = data_io.load_swbd_same_diff(rng, options_dict["data_dir"])
    train_x, train_matches_vec, train_labels = datasets[0]
    dev_x, dev_matches_vec, dev_labels = datasets[1]
    test_x, test_matches_vec, test_labels = datasets[2]

    # Flatten data
    d_in = 39*200
    train_x = train_x.reshape((-1, d_in))
    dev_x = dev_x.reshape((-1, d_in))
    test_x = test_x.reshape((-1, d_in))

    # Make batch iterators
    train_batch_iterator = BatchIteratorSameDifferent(
        rng, train_matches_vec, options_dict["batch_size"],
        n_same_pairs=options_dict["n_same_pairs"], sample_diff_every_epoch=True
        )
    validate_batch_iterator = BatchIteratorSameDifferent(
        rng, dev_matches_vec, options_dict["batch_size"],
        n_same_pairs=options_dict["n_same_pairs"],
        sample_diff_every_epoch=False
        )
    test_batch_iterator = BatchIteratorSameDifferent(
        rng, test_matches_vec, options_dict["batch_size"],
        n_same_pairs=options_dict["n_same_pairs"],
        sample_diff_every_epoch=False
        )


    # Setup model

    logger.info("Building Siamese CNN")

    # Symbolic variables
    y = T.ivector("y")      # indicates whether x1 and x2 is same (1) or different (0)
    x1 = T.matrix("x1")
    x2 = T.matrix("x2")
    x1_indices = T.ivector("x1_indices")
    x2_indices = T.ivector("x2_indices")

    # Build model
    input_shape = (options_dict["batch_size"], 1, 39, 200)
    model = siamese.SiameseCNN(
        rng, x1, x2, input_shape,
        conv_layer_specs=options_dict["conv_layer_specs"],
        hidden_layer_specs=options_dict["hidden_layer_specs"],
        srng=srng,
        dropout_rates=options_dict["dropout_rates"],
        )
    if options_dict["loss"] == "cos_cos2":
        if options_dict["dropout_rates"] is not None:
            loss = model.dropout_loss_cos_cos2(y)
        else:
            loss = model.loss_cos_cos2(y)
        error = model.loss_cos_cos2(y)  # doesn't include regularization or dropout
    elif options_dict["loss"] == "cos_cos":
        if options_dict["dropout_rates"] is not None:
            loss = model.dropout_loss_cos_cos(y)
        else:
            loss = model.loss_cos_cos(y)
        error = model.loss_cos_cos(y)
    elif options_dict["loss"] == "cos_cos_margin":
        if options_dict["dropout_rates"] is not None:
            loss = model.dropout_loss_cos_cos_margin(y)
        else:
            loss = model.loss_cos_cos_margin(y)
        error = model.loss_cos_cos_margin(y)
    elif options_dict["loss"] == "euclidean_margin":
        if options_dict["dropout_rates"] is not None:
            loss = model.dropout_loss_euclidean_margin(y)
        else:
            loss = model.loss_euclidean_margin(y)
        error = model.loss_euclidean_margin(y)
    else:
        assert False, "Invalid loss: " + options_dict["loss"]

    # Add regularization
    if options_dict["l1_weight"] > 0. or options_dict["l2_weight"] > 0.:
        loss = loss + options_dict["l1_weight"]*model.l1 + options_dict["l2_weight"]* model.l2

    # Compile test functions
    same_distance = model.cos_same(y)  # track the distances of same and different pairs separately
    diff_distance = model.cos_diff(y)
    outputs = [error, loss, same_distance, diff_distance]
    theano_mode = theano.Mode(linker="cvm")
    test_model = theano.function(
        inputs=[x1_indices, x2_indices, y],
        outputs=outputs,
        givens={
            x1: test_x[x1_indices],
            x2: test_x[x2_indices],
            },
        mode=theano_mode,
        )
    validate_model = theano.function(
        inputs=[x1_indices, x2_indices, y],
        outputs=outputs,
        givens={
            x1: dev_x[x1_indices],
            x2: dev_x[x2_indices],
            },
        mode=theano_mode,
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
        inputs=[x1_indices, x2_indices, y],
        outputs=outputs,
        updates=updates,
        givens={
            x1: train_x[x1_indices],
            x2: train_x[x2_indices],
            },
        mode=theano_mode,
        )


    # Train model

    logger.info("Training Siamese CNN")
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
    layers_output_dict = apply_layers.apply_layers(options_dict["model_dir"], "dev", batch_size=645)  # batch size covers 10965 out of 10966 tokens
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


def load_siamese_cnn(options_dict):

    model_fn = path.join(options_dict["model_dir"], "model.pkl.gz")

    # Symbolic variables
    x1 = T.matrix("x1")
    x2 = T.matrix("x2")

    # Random number generators
    rng = np.random.RandomState(options_dict["rnd_seed"])

    # Build model
    input_shape = (options_dict["batch_size"], 1, 39, 200)
    model = siamese.SiameseCNN(
        rng, x1, x2, input_shape,
        conv_layer_specs=options_dict["conv_layer_specs"],
        hidden_layer_specs=options_dict["hidden_layer_specs"],
        dropout_rates=None,  # dropout is not performed after training
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
    train_siamese_cnn(options_dict)


if __name__ == "__main__":
    main()
