"""
Functions for dealing with data input and output.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from os import path
import cPickle as pickle
import gzip
import logging
import numpy as np
import struct
import theano
import theano.tensor as T

import samediff

logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------#
#                            GENERAL I/O FUNCTIONS                            #
#-----------------------------------------------------------------------------#

def smart_open(filename, mode=None):
    """Opens a file normally or using gzip based on the extension."""
    if path.splitext(filename)[-1] == ".gz":
        if mode is None:
            mode = "rb"
        return gzip.open(filename, mode)
    else:
        if mode is None:
            mode = "r"
        return open(filename, mode)


def read_kaldi_ark_from_scp(scp_fn, ark_base_dir=""):
    """
    Read a binary Kaldi archive and return a dict of Numpy matrices, with the
    utterance IDs of the SCP as keys. Based on the code:
    https://github.com/yajiemiao/pdnn/blob/master/io_func/kaldi_feat.py

    Parameters
    ----------
    ark_base_dir : str
        The base directory for the archives to which the SCP points.
    """

    ark_dict = {}

    with open(scp_fn) as f:
        for line in f:
            if line == "":
                continue
            utt_id, path_pos = line.replace("\n", "").split(" ")
            ark_path, pos = path_pos.split(":")

            ark_path = path.join(ark_base_dir, ark_path)

            ark_read_buffer = smart_open(ark_path, "rb")
            ark_read_buffer.seek(int(pos),0)
            header = struct.unpack("<xcccc", ark_read_buffer.read(5))
            assert header[0] == "B", "Input .ark file is not binary"

            rows = 0
            cols= 0
            m, rows = struct.unpack("<bi", ark_read_buffer.read(5))
            n, cols = struct.unpack("<bi", ark_read_buffer.read(5))

            tmp_mat = np.frombuffer(ark_read_buffer.read(rows*cols*4), dtype=np.float32)
            utt_mat = np.reshape(tmp_mat, (rows, cols))

            ark_read_buffer.close()

            ark_dict[utt_id] = utt_mat

    return ark_dict


#-----------------------------------------------------------------------------#
#                         THEANO DATASET I/O FUNCTIONS                        #
#-----------------------------------------------------------------------------#

def load_swbd_labelled(rng, data_dir, min_count=1):
    """
    Load the Switchboard data with their labels.

    Only tokens that occur at least `min_count` times in the training set
    is considered.
    """

    def get_data_and_labels(set):

        npz_fn = path.join(data_dir, "swbd." + set + ".npz")
        logger.info("Reading: " + npz_fn)

        # Load data and shuffle
        npz = np.load(npz_fn)
        utts = sorted(npz.keys())
        rng.shuffle(utts)
        x = [npz[i] for i in utts]

        # Get labels for each utterance
        labels = swbd_utts_to_labels(utts)

        return x, labels

    train_x, train_labels = get_data_and_labels("train")
    dev_x, dev_labels = get_data_and_labels("dev")
    test_x, test_labels = get_data_and_labels("test")

    logger.info("Finding types with at least " + str(min_count) + " tokens")

    # Determine the types with the minimum count
    type_counts = {}
    for label in train_labels:
        if not label in type_counts:
            type_counts[label] = 0
        type_counts[label] += 1
    min_types = set()
    i_type = 0
    word_to_i_map = {}
    for label in type_counts:
        if type_counts[label] >= min_count:
            min_types.add(label)
            word_to_i_map[label] = i_type
            i_type += 1

    # Filter the sets
    def filter_set(x, labels):
        filtered_x = []
        filtered_i_labels = []
        for cur_x, label in zip(x, labels):
            if label in word_to_i_map:
                filtered_x.append(cur_x)
                filtered_i_labels.append(word_to_i_map[label])
        return filtered_x, filtered_i_labels
    train_x, train_labels = filter_set(train_x, train_labels)
    dev_x, dev_labels = filter_set(dev_x, dev_labels)
    test_x, test_labels = filter_set(test_x, test_labels)

    # Convert to shared variables
    def shared_dataset(x, y, borrow=True):
        shared_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, "int32")
    train_x, train_y = shared_dataset(train_x, train_labels)
    dev_x, dev_y = shared_dataset(dev_x, dev_labels)
    test_x, test_y = shared_dataset(test_x, test_labels)

    return [(train_x, train_y), (dev_x, dev_y), (test_x, test_y)], word_to_i_map



def load_swbd_same_diff(rng, data_dir):

    logger.info("Loading same and different pairs: " + data_dir)

    datasets = []

    for set in ["train", "dev", "test"]:

        npz_fn = path.join(data_dir, "swbd." + set + ".npz")
        logger.info("Reading: " + npz_fn)

        # Load data and shuffle
        npz = np.load(npz_fn)
        utt_ids = sorted(npz.keys())
        rng.shuffle(utt_ids)
        x = [npz[i] for i in utt_ids]

        # Get labels for each utterance
        labels = swbd_utts_to_labels(utt_ids)

        matches_vec = samediff.generate_matches_array(labels)
        shared_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)

        # Create a tuple for this set and add to `data_sets`
        datasets.append((shared_x, matches_vec, labels))

    return datasets


# def load_swbd_wordsim_old(rng, input_dir="input"):

#     logger.info("Loading Switchboard word-similarity data from directory: " + input_dir)

#     datasets = []

#     for set in ["train", "dev", "test"]:

#         logger.info("Loading " + set + " set from " + input_dir)

#         # Filenames
#         basename = "swbd." + set 
#         npz_fn = path.join(input_dir, basename + ".npz")
#         keys_fn = path.join(input_dir, basename + ".keys")
#         same_pairs_fn = path.join(input_dir, basename + ".same.pairs.indices")
#         diff_pairs_fn = path.join(input_dir, basename + ".diff.pairs.indices")

#         # Read Numpy archive in order of keys
#         with open(keys_fn) as f:
#             keys = [i.strip() for i in f]
#         npz = np.load(npz_fn)
#         x = [npz[i] for i in keys]

#         # Combine same and different pair indices and construct y
#         x1_indices = []     # indices of first token in pair
#         x2_indices = []     # indices of second token in pair
#         y = []      # 1 if pair is same, 0 if pair is different

#         # Read same pair indices
#         with open(same_pairs_fn) as f:
#             for line in f:
#                 i, j = line.strip().split(" ")
#                 x1_indices.append(int(i))
#                 x2_indices.append(int(j))
#                 y.append(1)  # these are all the same

#         # Read different pair indices
#         with open(diff_pairs_fn) as f:
#             for line in f:
#                 i, j = line.strip().split(" ")
#                 x1_indices.append(int(i))
#                 x2_indices.append(int(j))
#                 y.append(0)  # these are all different


#         # Shuffle x1_indices, x2_indices, and y in the same order
#         x1_indices = np.asarray(x1_indices, dtype=theano.config.floatX)
#         x2_indices = np.asarray(x2_indices, dtype=theano.config.floatX)
#         y = np.asarray(y, dtype=theano.config.floatX)
#         shuffle_order = np.arange(len(y))
#         rng.shuffle(shuffle_order)
#         x1_indices = x1_indices[shuffle_order]
#         x2_indices = x2_indices[shuffle_order]
#         y = y[shuffle_order]

#         # Create shared variables
#         shared_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)
#         # shared_x1_indices = theano.shared(x1, borrow=True)
#         # shared_x2_indices = theano.shared(x2, borrow=True)
#         shared_y = theano.shared(y, borrow=True)

#         # Create a tuple for this set and add to `data_sets`
#         datasets.append((
#             shared_x,
#             np.asarray(x1_indices, np.int32),
#             np.asarray(x2_indices, np.int32),
#             np.asarray(y, np.int32)
#             # T.cast(shared_x1_indices, "int32"),
#             # T.cast(shared_x2_indices, "int32"),
#             # T.cast(shared_y, "int32")
#             ))

#         # print shared_x.eval()
#         # print datasets[-1][-3].eval()
#         # print datasets[-1][-2].eval()
#         # print datasets[-1][-1].eval()

#     return datasets


# def load_swbd_labelled(pkl_fn):

#     logger.info("Loading Switchboard labelled data: " + pkl_fn)
#     f = smart_open(pkl_fn)
#     train_mean = pickle.load(f)
#     train_feats = pickle.load(f)
#     train_labels = pickle.load(f)
#     test_feats = pickle.load(f)
#     test_labels = pickle.load(f)
#     dev_feats = pickle.load(f)
#     dev_labels = pickle.load(f)
#     f.close()

#     def shared_dataset(x, y, borrow=True):
#         shared_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=borrow)
#         shared_y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=borrow)
#         return shared_x, T.cast(shared_y, "int32")

#     train_x, train_y = shared_dataset(train_feats, train_labels)
#     test_x, test_y = shared_dataset(test_feats, test_labels)
#     dev_x, dev_y = shared_dataset(dev_feats, dev_labels)

#     return [(train_x, train_y), (dev_x, dev_y), (test_x, test_y)]


# def load_npy(npy_fn):
#     x = np.load(npy_fn)
#     shared_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)
#     return shared_x


def load_npz(npz_fn):
    logger.info("Reading: " + npz_fn)
    npz = np.load(npz_fn)
    keys = sorted(npz.keys())
    x = [npz[i] for i in keys]
    shared_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)
    return shared_x, keys


# def load_mnist(pkl_gz_fn):
#     """
#     Load the MNIST dataset; the data is downloaded if the file `pkl_gz_fn` does
#     not exist.
#     """

#     if not path.isfile(pkl_gz_fn):
#         import urllib
#         origin = "http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz"
#         logger.info("Dowloading data from: %s" % origin)
#         urllib.urlretrieve(origin, pkl_gz_fn)

#     logger.info("Loading MNIST data")
#     # train_set, dev_set, test_set: (input, target)
#     # `input` is an array of 2 dimensions which row's correspond to an example.
#     # `target` is a vector that have the same length as the number of rows in
#     # the input. It should give the target to the example with the same index
#     # in the input.
#     f = gzip.open(pkl_gz_fn, "rb")
#     train_set, dev_set, test_set = pickle.load(f)
#     f.close()

#     def shared_dataset(data_xy, borrow=True):
#         """Load data into shared variables, for efficient GPU access."""
#         data_x, data_y = data_xy
#         shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
#         shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)

#         # The T.cast returns a function, which allows `shared_y` to still be
#         # stored as floatX (for use on the GPUs), while using the integer
#         # values for the actual evaluation.
#         return shared_x, T.cast(shared_y, "int32")

#     test_set_x, test_set_y = shared_dataset(test_set)
#     dev_set_x, dev_set_y = shared_dataset(dev_set)
#     train_set_x, train_set_y = shared_dataset(train_set)

#     return [(train_set_x, train_set_y), (dev_set_x, dev_set_y), (test_set_x, test_set_y)]


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def swbd_utt_to_label(utt):
    return "_".join(utt.split("_")[:-2])


def swbd_utts_to_labels(utts):
    labels = []
    for utt in utts:
        labels.append(swbd_utt_to_label(utt))
    return labels


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    logging.basicConfig(level=logging.DEBUG)

    rng = np.random.RandomState(42)
    # load_swbd_same_diff(rng, "input/tmp")
    # load_swbd_wordsim(rng, "icassp", "input_tmp")

    # ark = read_kaldi_ark_from_scp(
    #     "/share/data/speech-multiview/kamperh/kaldi/swbd/s5c.herman/data/clusters_gt/clusters_gt.mfcc.min_count_5.scp",
    #     ark_base_dir="/share/data/speech-multiview/kamperh/kaldi/swbd/s5c.herman/"
    #     )
    # print ark["individuals_sw03105-B_029898-029978"]
    # print ark["individuals_sw03105-B_029898-029978"].shape


if __name__ == "__main__":
    main()


