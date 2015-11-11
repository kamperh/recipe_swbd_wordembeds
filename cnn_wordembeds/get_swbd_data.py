#!/usr/bin/env python

"""
Get the Switchboard data in Numpy archive format.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from os import path
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL.Image as Image
import sys

sys.path.append(path.join("..", "couscous"))

from couscous import plotting
from data_io import read_kaldi_ark_from_scp, smart_open

data_base_dir = "/share/data/speech-multiview/kamperh/kaldi/swbd/s5c.herman"
n_padded = 200  # pad to have this many feature vectors in every instance


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("data_dir", type=str, help="directory to write data to")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def pad_images_width(images, width_padded, pad_value=0.):
    n_images = len(images)
    height = images[0].shape[0]
    padded_images = np.ones((n_images, height, width_padded))*pad_value
    for i_data in xrange(n_images):
        width = images[i_data].shape[1]
        padding = int(np.round((width_padded - width) / 2.))
        padded_images[i_data, :, padding:padding + width] = images[i_data]
    return padded_images


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()
    
    if not path.isdir(args.data_dir):
        os.makedirs(args.data_dir)

    print datetime.datetime.now()


    # Training data: ICASSP 2015

    scp_fn = path.join(data_base_dir, "data/clusters_gt/clusters_gt.mfcc.scp")
    print "Reading:", scp_fn
    ark_dict = read_kaldi_ark_from_scp(scp_fn, data_base_dir)

    ids = sorted(ark_dict.keys())
    mats = [ark_dict[i].T for i in ids]
    train_mean = np.mean(np.hstack(mats).flatten())
    mats = pad_images_width(mats, n_padded, train_mean) - train_mean
    print "Padded and normalized", mats.shape[0], "data instances"

    npz_fn =  path.join(args.data_dir, "swbd.train.npz")
    print "Writing:", npz_fn
    npz_dict = {}
    for i, utt_id in enumerate(ids):
        npz_dict[utt_id] = mats[i]
    np.savez_compressed(npz_fn, **npz_dict)

    train_mean_fn = path.join(args.data_dir, "train.mean")
    print "Writing:", train_mean_fn
    with open(train_mean_fn, "w") as f:
        f.write(str(train_mean) + "\n")

    plot_fn = path.join(args.data_dir, "train_example.png")
    print "Saving:", plot_fn
    image = Image.fromarray(plotting.array_to_pixels(mats[0]))
    image.save(plot_fn)


    # # Training data: ASRU 2013

    # scp_fn = path.join(data_base_dir, "data/clusters_gt_asru13/clusters_gt_asru13.mfcc.scp")
    # print "Reading:", scp_fn
    # ark_dict = read_kaldi_ark_from_scp(scp_fn, data_base_dir)

    # ids = sorted(ark_dict.keys())
    # mats = [ark_dict[i].T for i in ids]
    # train_mean_asru13 = np.mean(np.hstack(mats).flatten())
    # mats = pad_images_width(mats, n_padded, train_mean_asru13) - train_mean_asru13
    # print "Padded and normalized", mats.shape[0], "data instances"

    # npz_fn =  path.join(args.data_dir, "swbd.train_asru13.npz")
    # print "Writing:", npz_fn
    # npz_dict = {}
    # for i, utt_id in enumerate(ids):
    #     npz_dict[utt_id] = mats[i]
    # np.savez_compressed(npz_fn, **npz_dict)

    # train_mean_fn = path.join(args.data_dir, "train_asru13.mean")
    # print "Writing:", train_mean_fn
    # with open(train_mean_fn, "w") as f:
    #     f.write(str(train_mean_asru13) + "\n")

    # plot_fn = path.join(args.data_dir, "train_asru13_example.png")
    # print "Saving:", plot_fn
    # image = Image.fromarray(plotting.array_to_pixels(mats[0]))
    # image.save(plot_fn)


    # Test data

    scp_fn = path.join(data_base_dir, "data/samediff_test/samediff_test.mfcc.scp")
    print "Reading:", scp_fn
    ark_dict = read_kaldi_ark_from_scp(scp_fn, data_base_dir)

    ids = sorted(ark_dict.keys())
    mats = [ark_dict[i].T for i in ids]
    mats = pad_images_width(mats, n_padded, train_mean) - train_mean
    print "Padded and normalized", mats.shape[0], "data instances"

    npz_fn =  path.join(args.data_dir, "swbd.test.npz")
    print "Writing:", npz_fn
    npz_dict = {}
    for i, utt_id in enumerate(ids):
        npz_dict[utt_id] = mats[i]
    np.savez_compressed(npz_fn, **npz_dict)


    # Development data

    scp_fn = path.join(data_base_dir, "data/samediff_dev/samediff_dev.mfcc.scp")
    print "Reading:", scp_fn
    ark_dict = read_kaldi_ark_from_scp(scp_fn, data_base_dir)

    ids = sorted(ark_dict.keys())
    mats = [ark_dict[i].T for i in ids]
    mats = pad_images_width(mats, n_padded, train_mean) - train_mean
    print "Padded and normalized", mats.shape[0], "data instances"

    npz_fn =  path.join(args.data_dir, "swbd.dev.npz")
    print "Writing:", npz_fn
    npz_dict = {}
    for i, utt_id in enumerate(ids):
        npz_dict[utt_id] = mats[i]
    np.savez_compressed(npz_fn, **npz_dict)


    print datetime.datetime.now()


if __name__ == "__main__":
    main()
