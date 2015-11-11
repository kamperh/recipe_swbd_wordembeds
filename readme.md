Recipe: Acoustic Word Embeddings for Switchboard
================================================


Overview
--------

This is a recipe for extracting acoustic word embeddings for a subset of the
Switcboard corpus. The models are described in detail in:

- H. Kamper, W. Wang, and K. Livescu, "Deep convolutional acoustic word
  embeddings using word-pair side information," arXiv preprint
  arXiv:1510.01032, 2015.

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
