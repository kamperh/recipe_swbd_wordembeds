#!/usr/bin/env python

"""
Create the word-pair segments file used for imposing weak top-down constraints.

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
    parser.add_argument("clusters_fn", type=str, help="original file list")
    parser.add_argument("segments_fn", type=str, help="output segments file")
    parser.add_argument("--feats_scp_fn", type=str, help="(default: %(default)s)", default="data/train/feats.scp")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    wordpairs_fn = args.clusters_fn
    feats_scp_fn = args.feats_scp_fn
    segments_fn = args.segments_fn

    print datetime.datetime.now()

    # Create utterance segments dict
    utterance_segs = {}  # utterance_segs["sw02001-A_000098-001156"] is (98, 1156)
    for line in open(feats_scp_fn):
        line = line.split(" ")[0]
        utterance_segs[line] = tuple([int(i) for i in line.split("_")[-1].split("-")])

    # Create word-pair segments dict
    word_segs = {}  # word_segs["organized_sw02111-A_000280_000367"] is ("sw02111-A", 280, 367)
    for line in open(wordpairs_fn):

        # Extract info
        word, conversation_1, speaker_id_1, start_1, end_1, conversation_2, speaker_id_2, start_2, end_2 = (
            line.split(" ")
            )

        # Add first word in pair
        conversation_1 = "sw0" + conversation_1[:-1] + "-" + conversation_1[-1]
        start_1 = int(start_1)
        end_1 = int(end_1)
        word_id_1 = word + "_" + conversation_1 + "_" + "%06d" % start_1 + "-" + "%06d" % end_1
        if word_id_1 not in word_segs:
            word_segs[word_id_1] = (conversation_1, start_1, end_1)

        # Add second word in pair
        conversation_2 = "sw0" + conversation_2[:-1] + "-" + conversation_2[-1]
        start_2 = int(start_2)
        end_2 = int(end_2)
        word_id_2 = word + "_" + conversation_2 + "_" + "%06d" % start_2 + "-" + "%06d" % end_2
        if word_id_2 not in word_segs:
            word_segs[word_id_2] = (conversation_2, start_2, end_2)

    # Write segments file
    f = open(segments_fn, "w")
    print "Writing segments to: " + segments_fn
    i_word = 0
    for word_id in word_segs:
        conversation, word_start, word_end = word_segs[word_id]
        for utt_id in [i for i in utterance_segs.keys() if i.startswith(conversation)]:
            utt_start, utt_end = utterance_segs[utt_id]
            if word_start > utt_start and word_start < utt_end:
                start = word_start - utt_start - 1  # one extra frame at start (i.e. 15 ms overlap of window)
                if start < 0:
                    start = 0
                # end = word_end - utt_start - 3 + 1
                end = word_end - utt_start - 2 + 1  # also corresponds to a frame with 15 ms overlap 
                if end > utt_end:
                    end = utt_end
                f.write(word_id + " " + utt_id + " " + str(start) + " " + str(end) + "\n")
                i_word += 1
                # print "Processed " + str(i_word) + " words."
    f.close()

    print datetime.datetime.now()


if __name__ == "__main__":
    main()
