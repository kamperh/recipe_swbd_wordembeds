#!/bin/bash
# Herman Kamper, kamperh@gmail.com, 2015.
# Based loosely an parts of train_mono.sh.

nj=4
cmd=run.pl

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
    echo "usage: ${0} data_dir exp_dir feat_dir"
    exit 1;
fi

data=$1
dir=$2
mfccdir=$3

name=`basename $data`

mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

feats="apply-cmvn --norm-vars=true --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark,scp:$mfccdir/mfcc_cmnv_dd_$name.JOB.ark,$mfccdir/mfcc_cmnv_dd_$name.JOB.scp"

$train_cmd JOB=1:$nj $dir/log/cmvn_dd.JOB.log $feats || exit 1;
