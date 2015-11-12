#!/bin/bash
# Herman Kamper, kamperh@gmail.com, 2015.

# This script uses some of the scripts in the Switcboard Kaldi recipe
# (kaldi-trunk/egs/swbd/s5c). The steps/ and utils/ directories are linked to
# the WSJ Kaldi directories (kaldi-trunk/egs/wsj/s5).

set -e

. cmd.sh
. path.sh


# Make links and get SCP files for switchboard
local/swbd1_data_download.sh /share/data/speech/Datasets/switchboard/
local/swbd1_data_prep.sh /share/data/speech/Datasets/switchboard/

# Get MFCC features
mfccdir=mfcc
x=train
steps/make_mfcc.sh --nj 50 --cmd "$train_cmd" data/$x exp/make_mfcc/$x $mfccdir
steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
utils/fix_data_dir.sh data/$x
local/cmvn_dd.sh --nj 50 --cmd "$train_cmd" data/train exp/make_cmvn_dd/$x $mfccdir
rm $mfccdir/raw_mfcc_train.*  # clean raw files, won't need these
cat $mfccdir/mfcc_cmnv_dd_train.*.scp > data/train/mfcc_cmvn_dd.scp
mfcc_scp=data/train/mfcc_cmvn_dd.scp

# Prepare clusters_gt set, used in [Kamper et al., ICASSP 2015]
x=clusters_gt
mkdir -p data/${x}
local/make_clusters_segments.py ../data/wordpairs_samtrain.list data/${x}/segments
extract-rows data/${x}/segments "scp:$mfcc_scp" "ark,scp:data/${x}/${x}.mfcc.ark,data/${x}/${x}.mfcc.scp"

# Prepare samediff_dev
x=samediff_dev
mkdir -p data/${x}
local/make_dev_segments.py ../data/devset.list data/${x}/segments
extract-rows data/${x}/segments "scp:$mfcc_scp" "ark,scp:data/${x}/${x}.mfcc.ark,data/${x}/${x}.mfcc.scp"

# Prepare samediff_test
x=samediff_test
mkdir -p data/${x}
local/make_test_segments.py ../data/words_gamtrain_lc.list data/${x}/segments
extract-rows data/${x}/segments "scp:$mfcc_scp" "ark,scp:data/${x}/${x}.mfcc.ark,data/${x}/${x}.mfcc.scp"
