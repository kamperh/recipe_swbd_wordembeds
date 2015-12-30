Recipe: Acoustic Word Embeddings for Switchboard
================================================


Overview
--------

This is a recipe for extracting acoustic word embeddings for a subset of the
Switchboard corpus. The models are described in detail in [Kamper et al.,
2015](http://arxiv.org/abs/1510.01032):

- H. Kamper, W. Wang, and K. Livescu, "Deep convolutional acoustic word
  embeddings using word-pair side information," Proc. ICASSP, 2016.

Please cite this paper if you use this code. All the neural networks are
implemented in the package [couscous](https://github.com/kamperh/couscous).


Steps
-----

1.  Install all dependencies (below).

2.  Clone [couscous](https://github.com/kamperh/couscous) into the appropriate
    directory:

        mkdir ../src
        git clone https://github.com/kamperh/couscous.git ../src/couscous

3.  Run the steps in [kaldi_features/run.sh](kaldi_features/run.sh).

4.  Run the steps in [cnn_wordembeds/readme.md](cnn_wordembeds/readme.md).

5.  If you run the steps correctly above, then if you execute the following:

        cd cnn_wordembeds/
        ./apply_layers.py models/siamese_triplets_cnn.1/ test
        ./eval_samediff.py \
            models/siamese_triplets_cnn.1/swbd.test.layer_-1.npz

    Then the evaluation should show the following output:

        Average precision: 0.537404372048
        Precision-recall breakeven: 0.542724052097

    The average precision (AP) of 0.537 is used for the number reported in
    Table 1, row 9 of [Kamper et al., 2015](http://arxiv.org/abs/1510.01032).


Dependencies
------------

- [Kaldi](http://kaldi.sourceforge.net/)
- [Theano](http://deeplearning.net/software/theano/) and all its dependencies.
- [couscous](https://github.com/kamperh/couscous): should be cloned into the
  directory `../src/couscous`.


Collaborators
-------------

- [Herman Kamper](http://www.kamperh.com/)
- [Weiran Wang](http://ttic.uchicago.edu/~wwang5/)
- [Karen Livescu](http://ttic.uchicago.edu/~klivescu/)
