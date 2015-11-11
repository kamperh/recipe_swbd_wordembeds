#!/usr/bin/env python

"""
Perform same-different evaluation of fixed-dimensional representations.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from scipy.spatial.distance import pdist
import argparse
import datetime
import numpy as np
import sys

import samediff


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("npz_fn", type=str, help="Numpy archive")
    parser.add_argument(
        "--metric", choices=["cosine", "euclidean", "hamming", "chebyshev", "kl"], default="cosine",
        help="distance metric (default: %(default)s)"
        )
    # parser.add_argument(
    #     "--normalize", dest="normalize", action="store_true",
    #     help="normalize embeddings to unit sphere before calculating distances (default is not to do this)"
    #     )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print datetime.datetime.now()

    print "Reading:", args.npz_fn
    npz = np.load(args.npz_fn)

    print datetime.datetime.now()

    # if args.normalize:
    #     print "Normalizing embeddings"
    # else:
    print "Ordering embeddings"
    n_embeds = 0
    X = []
    ids = []
    for label in sorted(npz):
        ids.append(label)
        X.append(npz[label])
        n_embeds += 1
    X = np.array(X)
    print "No. embeddings:", n_embeds
    print "Embedding dimensionality:", X.shape[1]

    print datetime.datetime.now()

    print "Calculating distances"
    metric = args.metric
    if metric == "kl":
        import scipy.stats
        metric = scipy.stats.entropy
    distances = pdist(X, metric=metric)

    print "Getting labels"
    labels = []
    for utt_id in ids:
        word = "_".join(utt_id.split("_")[:-2])
        labels.append(word)

    print "Calculating average precision"
    matches = samediff.generate_matches_array(labels)

    ap, prb = samediff.average_precision(distances[matches == True], distances[matches == False])
    print "Average precision:", ap
    print "Precision-recall breakeven:", prb

    print datetime.datetime.now()


if __name__ == "__main__":
    main()
