Acoustic Word Embeddings on Switchboard using CNNs
==================================================


Data preparation
----------------

Get the Switchboard data used in [Kamper et al., ICASSP 2015], and pad all the
word segments to 200 frames (2 seconds):

    ./get_swbd_data.py data/icassp15.0


