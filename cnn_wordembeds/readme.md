Acoustic Word Embeddings on Switchboard using CNNs
==================================================


Data preparation
----------------

Get the Switchboard data used in [Kamper et al., 2015](http://www.kamperh.com
/papers/kamper+jansen+goldwater_interspeech2015.pdf), and pad all the word
segments to 200 frames (2 seconds):

    ./get_swbd_data.py data/icassp15.0


Word classifier networks
------------------------

Train and evaluate a MLP word classification network:

    ./train_mlp.py models/mlp.1
    ./apply_layers.py models/mlp.1 dev
    ./eval_samediff.py models/mlp.1/swbd.dev.layer_-1.npz

Analyze the network and the produced embeddings:
    
    ./analyze_mlp.py models/mlp.1/
    ./analyze_embeds.py --normalize --word_type \
        absolutely,doctor,doctors,particular,particularly,quality,recycling \
        models/mlp.1/swbd.dev.layer_-1.npz

Train and evaluate a CNN word classification network:

    ./train_cnn.py models/cnn.1
    ./apply_layers.py models/cnn.1 dev
    ./eval_samediff.py models/cnn.1/swbd.dev.layer_-1.npz

Analyze the network and the produced embeddings:

    ./analyze_cnn.py models/cnn.1
    ./analyze_embeds.py --normalize --word_type \
        absolutely,doctor,doctors,particular,particularly,quality,recycling \
        models/cnn.1/swbd.dev.layer_-1.npz


Word similarity networks
------------------------

Train and evaluate a Siamese CNN word similarity network:

    ./train_siamese_cnn.py models/siamese_cnn.1
    ./apply_layers.py models/siamese_cnn.1/ dev
    ./eval_samediff.py models/mlp.1/swbd.dev.layer_-1.npz

Analyze the network and the produced embeddings:
    
    ./analyze_cnn.py models/siamese_cnn.1/
    ./analyze_embeds.py --normalize --word_type \
        absolutely,doctor,doctors,particular,particularly,quality,recycling \
        models/siamese_cnn.1/swbd.dev.layer_-1.npz

Train and evaluate a Siamese triplets CNN word similarity network:

    ./train_siamese_triplets_cnn.py models/siamese_triplets_cnn.1
    ./apply_layers.py models/siamese_triplets_cnn.1/ dev
    ./eval_samediff.py models/siamese_triplets_cnn.1/swbd.dev.layer_-1.npz

Analyze the network and the produced embeddings:

    ./analyze_cnn.py models/siamese_triplets_cnn.1/
    ./analyze_embeds.py --normalize --word_type \
        absolutely,doctor,doctors,particular,particularly,quality,recycling \
        models/siamese_triplets_cnn.1/swbd.dev.layer_-1.npz


Optimizing hyper-parameters
---------------------------

For most of the models, there is a corresponding "sweep" script. For example,
to sweep the options for the CNN word classifier, run the following (edit the
scripts to sweep different sets of options):

    ./train_cnn_sweep.py models/cnn_sweep.1/
    ./analyze_sweep.py models/cnn_sweep.1/
