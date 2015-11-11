#!/usr/bin/env python

"""
Analyze a given file with embedding tokens.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from os import path
from sklearn import decomposition, ensemble, manifold
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

basedir = path.dirname(path.abspath(__file__))
sys.path.append(path.join(basedir, "..", "couscous"))

from couscous import plotting
from data_io import swbd_utt_to_label


#-----------------------------------------------------------------------------#
#                                   PLOTTING                                  #
#-----------------------------------------------------------------------------#

def plot_raw_embeds(npz, types=None):
    """Plot all the embeddings of type `types`, if None plot everything."""

    # Get embeddings
    embeddings = []
    labels = []
    for key in npz:
        if "_" in key:
            label = "_".join(key.split("_")[:-2])
        else:
            label = key
        if types is None:
            labels.append(label)
            embeddings.append(npz[key])
        elif label in types:
            labels.append(label)
            embeddings.append(npz[key])
    n_embeds = len(embeddings)

    # Now sort by label
    sort_order = np.argsort(np.array(labels))
    sorted_labels = np.array(labels)[sort_order]

    # Get cluster tick positions
    type_ticks = [0]
    for i in range(len(sorted_labels) - 1):
        if sorted_labels[i] != sorted_labels[i + 1]:
            type_ticks.append(i + 1)
    type_ticks.append(n_embeds)

    # Get label positions and labels
    type_label_ticks = []
    type_labels = []
    for i in sorted(list(set(labels))):
        where = np.where(sorted_labels == i)[0]
        if len(where) == 0:
            continue
        pos = int(np.mean(where))
        type_label_ticks.append(pos)
        type_labels.append(i)

    # print "Plotting all embeddings"

    # Variables used for plotting
    labels_offset = 1.04
    par2_linewidth = 0.5

    fig, host = plt.subplots()
    par2 = host.twinx()
    par2.spines["right"].set_position(("axes", labels_offset))
    plotting.make_patch_spines_invisible(par2)
    par2.spines["right"].set_visible(True)
    par2.set_ylim([0, n_embeds])
    par2.invert_yaxis()
    par2.set_yticks(type_ticks)
    par2.set_yticklabels([])
    par2.tick_params(axis="y", width=par2_linewidth, length=10)
    par2.spines["right"].set_linewidth(par2_linewidth)
    par2.set_yticks(type_label_ticks, minor=True)
    par2.set_yticklabels(type_labels, minor=True)
    par2.set_ylabel("Word types")
    for line in par2.yaxis.get_minorticklines():
        line.set_visible(False)

    cax = host.imshow(np.array(embeddings)[sort_order], interpolation="nearest", aspect="auto")
    host.set_yticks([])
    # host.set_xticklabels([])
    host.set_ylabel("Word embedding vector")
    host.set_xlabel("Embedding dimensions")
    # fig.colorbar(cax, orientation="horizontal")


# From http://scikit-learn.org/stable/_downloads/plot_lle_digits.py.
def plot_embeds_2d(embeds_dict, types=None):
    print("Computing PCA projection")
    embeddings, labels = get_embeds_and_labels(embeds_dict, types)
    X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(embeddings)
    plot_data_labelled(X_pca, labels, "PCA")

    print("Computing t-SNE embedding")
    embeddings, labels = get_embeds_and_labels(embeds_dict, types)
    tsne = manifold.TSNE(n_components=2, perplexity=10, init="pca", random_state=0)
    X_tsne = tsne.fit_transform(embeddings)
    plot_data_labelled(X_tsne, labels, "t-SNE")

    # print("Computing Spectral embedding")
    # embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
    #                                       eigen_solver="arpack")
    # X_se = embedder.fit_transform(embeddings)
    # plot_data_labelled(X_se, labels)

    # print("Computing Totally Random Trees embedding")
    # hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
    #                                        max_depth=5)
    # X_transformed = hasher.fit_transform(embeddings)
    # pca = decomposition.TruncatedSVD(n_components=2)
    # X_reduced = pca.fit_transform(X_transformed)
    # plot_data_labelled(X_reduced, labels)

    # print("Computing MDS embedding")
    # clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
    # X_mds = clf.fit_transform(embeddings)
    # plot_data_labelled(X_mds, labels)

    print("Computing Isomap embedding")
    n_neighbors = 10
    X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(embeddings)
    plot_data_labelled(X_iso, labels, "Isomap (" + str(n_neighbors) + " neighbours)")


def plot_data_labelled(X, labels, title=None):
    ordered_labels = sorted(set(labels))
    n_labels = len(set(labels))

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(labels[i]),
                 color=plt.cm.Set1(1.0*ordered_labels.index(labels[i]) / n_labels),
                 fontdict={"weight": "bold", "size": 9})

    if title is not None:
        plt.title(title)

    plt.xticks([]), plt.yticks([])


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("npz_fn", type=str, help="")
    parser.add_argument("--word_type", type=str, help="show a plot for these word types, given as comma-seperated values")
    parser.add_argument("--plot_rnd", type=int, help="plot this number of randomly selected embeddings")
    parser.add_argument("--plot_all", action="store_true", help="plot all embeddings")
    parser.add_argument(
        "--normalize", dest="normalize", action="store_true",
        help="normalize embeddings to unit sphere before calculating distances (default is not to do this)"
        )
    parser.set_defaults(normalize=False)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def get_embeds_and_labels(embeds_dict, types=None):
    embeddings = []
    labels = []
    for utt in embeds_dict:
        if "_" in utt:
            label = swbd_utt_to_label(utt)
        else:
            label = utt
        if types is None:
            labels.append(label)
            embeddings.append(embeds_dict[utt])
        elif label in types:
            labels.append(label)
            embeddings.append(embeds_dict[utt])
    embeddings = np.array(embeddings)
    return embeddings, labels


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print "Reading:", args.npz_fn
    npz = np.load(args.npz_fn)

    if args.normalize:
        print "Normalizing embeddings"
        norm_npz = {}
        for key in npz:
            embed = npz[key]
            norm_npz[key] = embed/np.linalg.norm(embed)
        npz = norm_npz

    print "Minimum embedding value:", np.min([np.min(npz[key]) for key in npz])
    print "Maximum embedding value:", np.max([np.max(npz[key]) for key in npz])

    if args.word_type:
        if not "," in args.word_type:
            # A single word type
            print "Plotting embeddings for type:", args.word_type
            embeddings = []
            for key in npz:
                if args.word_type in key:
                    embed = npz[key]
                    embeddings.append(embed)
            print "No. embeddings matching type:", len(embeddings)
            plt.imshow(embeddings, interpolation="nearest", aspect="auto")
        else:
            # Multiple word types
            # plot_embeds_tsne(npz, args.word_type.split(","))
            plot_raw_embeds(npz, args.word_type.split(","))
            plot_embeds_2d(npz, args.word_type.split(","))

    # print "Example embedding:", npz[npz.keys()[0]]

    if args.plot_all:
        plot_raw_embeds(npz)
        # plot_embeds_2d(npz)

    if args.plot_rnd is not None:
        print "Analyzing", args.plot_rnd, "randomly sampled embeddings"
        random.seed(42)
        sample_keys = random.sample(npz.keys(), args.plot_rnd)
        npz_sampled = {}
        for key in sample_keys:
            npz_sampled[key] = npz[key]
        plot_raw_embeds(npz_sampled)
        plot_embeds_2d(npz_sampled)

    if args.word_type or args.plot_all or args.plot_rnd:
        plt.show()


if __name__ == "__main__":
    main()
