#!/usr/bin/env python

"""
Create segment file for devset word segments used in samediff evaluation.

Run from ../ directory.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2014-2015
"""

import argparse
import datetime
import os
import sys


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("devset_fn", type=str, help="original file list")
    parser.add_argument("segments_fn", type=str, help="output segments file")
    parser.add_argument("--feats_scp_fn", type=str, help="(default: %(default)s)", default="data/train/feats.scp")
    parser.add_argument(
        "--n_padding", default=0, type=int,
        help="the number of frames to pad with; i.e. the number of frames to add "
        "on either side of each segment (default: %(default)s)"
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

    devset_fn = args.devset_fn
    segments_fn = args.segments_fn
    feats_scp_fn = args.feats_scp_fn

    print datetime.datetime.now()
    print "n_padding:", args.n_padding

    # Create utterance segments dict
    utterance_segs = {}  # utterance_segs["sw02001-A_000098-001156"] is (98, 1156)
    for line in open(feats_scp_fn):
        line = line.split(" ")[0]
        utterance_segs[line] = tuple([int(i) for i in line.split("_")[-1].split("-")])

    # Create word segments dict
    word_segs = {}  # word_segs["organized_sw02111-A_000280_000367"] is ("sw02111-A", 280, 367)
    for line in open(devset_fn):
        word_id = line.strip()
        conversation = word_id[-23:-14]
        start = int(word_id[-13:-7])
        end = int(word_id[-6:])
        word_segs[word_id] = (conversation, start, end)

    # Write segments file
    f = open(segments_fn, "w")
    print "Writing segments to: " + segments_fn
    i_word = 0
    for word_id in word_segs:
        conversation, word_start, word_end = word_segs[word_id]
        for utt_id in [i for i in utterance_segs.keys() if i.startswith(conversation)]:
            utt_start, utt_end = utterance_segs[utt_id]
            # print utt_id
            if word_start > utt_start and word_start < utt_end:
                # Add one extra frame at start (i.e. 15 ms overlap of window) with additional padding
                start = word_start - utt_start - 1 - args.n_padding
                if start < 0:
                    start = 0
                # Also corresponds to a frame with 15 ms overlap with additional padding
                end = word_end - utt_start - 2 + 1  + args.n_padding
                if end > utt_end:
                    end = utt_end
                f.write(word_id + " " + utt_id + " " + str(start) + " " + str(end) + "\n")
                i_word += 1
                # print "Processed " + str(i_word) + " words."
    f.close()

    print datetime.datetime.now()


if __name__ == "__main__":
    main()
