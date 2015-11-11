"""
Sweeps options in a given options dictionary in multiple (GPU) processes.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from os import path
import copy
import hashlib
import itertools
import multiprocessing


def sweep_nn_options(default_options_dict, eval_function,
        sweep_options_dict={}, n_cnn_units=None, n_hidden_units=None,
        n_hidden_layers=None, n_hidden_units_final_layer=None, mode="serial",
        n_procs=None):
    """
    Apply `eval_function` by sweeping the options in `default_options_dict`.

    All models is written in separate directories to the `model_base_dir`
    directory.
    """

    assert mode in ["serial", "parallel"]

    # Find the elements which need to be swept
    sweep_options_list = sorted(sweep_options_dict.keys())
    sweep_options_value_list = [sweep_options_dict[option] for option in sweep_options_list]
    if n_cnn_units is not None:
        sweep_options_list.append("n_cnn_units")
        sweep_options_value_list.append(n_cnn_units)
    if n_hidden_units is not None:
        sweep_options_list.append("n_hidden_units")
        sweep_options_value_list.append(n_hidden_units)
    if n_hidden_layers is not None:
        assert len(default_options_dict["hidden_layer_specs"]) <= n_hidden_layers
        sweep_options_list.append("n_hidden_layers")
        sweep_options_value_list.append(n_hidden_layers)
    if n_hidden_units_final_layer is not None:
        sweep_options_list.append("n_hidden_units_final_layer")
        sweep_options_value_list.append(n_hidden_units_final_layer)

    # Build up a list of option dictionaries
    options_dict_list = []
    for cur_sweep_params_values in itertools.product(*sweep_options_value_list):
        
        # Copy static options
        cur_options_dict = copy.deepcopy(default_options_dict)

        # Add current dynamic options
        for i in xrange(len(sweep_options_list)):
            cur_option = sweep_options_list[i]
            cur_option_value = cur_sweep_params_values[i]
            if cur_option is "n_cnn_units":
                for i_layer, layer_spec in enumerate(cur_options_dict["conv_layer_specs"]):
                    cur_filter_shape = list(layer_spec["filter_shape"])
                    cur_filter_shape[0] = cur_option_value
                    if i_layer != 0:
                        cur_filter_shape[1] = cur_option_value
                    layer_spec["filter_shape"] = tuple(cur_filter_shape)
            elif cur_option is "n_hidden_units":
                for layer_spec in cur_options_dict["hidden_layer_specs"]:
                    layer_spec["units"] = cur_option_value
            elif cur_option is "n_hidden_layers":
                while len(cur_options_dict["hidden_layer_specs"]) < cur_option_value:
                    cur_options_dict["hidden_layer_specs"] = (
                        [cur_options_dict["hidden_layer_specs"][0]] +
                        cur_options_dict["hidden_layer_specs"]
                        )
            elif cur_option is "n_hidden_units_final_layer":
                cur_options_dict["hidden_layer_specs"][-1]["units"] = cur_option_value
            else:
                cur_options_dict[cur_option] = cur_option_value

        # Set output directory
        hasher = hashlib.md5(repr(sorted(cur_options_dict.items())).encode("ascii"))
        hash_str = hasher.hexdigest()[:10]
        cur_options_dict["model_dir"] = path.join(
            default_options_dict["model_dir"], hash_str
            )
        if path.isfile(path.join(cur_options_dict["model_dir"], "dev_ap.txt")):
            print "Skipping because of existing directory:", cur_options_dict["model_dir"]
            continue
        options_dict_list.append(cur_options_dict)
    print("No. of option sets: " + str(len(options_dict_list)))

    # Sweep dictionaries using the indicated `mode`
    if mode == "serial":
        # logging.basicConfig(level=logging.INFO)
        results = []
        for i, options_dict in enumerate(options_dict_list):
            result = eval_function(options_dict)
            results.append((options_dict, result))
    elif mode == "parallel":
        pool = multiprocessing.Pool(n_procs)
        results = []
        r = pool.map_async(eval_function, options_dict_list, callback=results.append)
        r.wait()
        r.get()
        print("No. of processes completed: " + str(len(results[0])) + " out of " + str(len(options_dict_list)))
        results = results[0]

    return results


def main():

    def eg_eval(x):
        return x["hidden_layer_specs"][0]["units"], len(x["hidden_layer_specs"])

    default_options_dict = {
        "data_dir": "data/icassp15.0",
        "model_dir": "models/tmp",
        # "data_dir": "data/tmp",
        "min_count": 5,             # minimum number of times a training label needs to occur
        "rnd_seed": 42,
        "batch_size": 30,
        "n_max_epochs": 30,
        "l1_weight": 0.0,
        "l2_weight": 0.0,
        "learning_rule": {
            "type": "adadelta",     # can be "momentum", "adadgrad", "adadelta"
            "rho": 0.9,             # parameters specific to learning rule
            "epsilon": 1e-6         
            },
        # "learning_rule": {
        #     "type": "momentum",
        #     "learning_rate": 0.01,
        #     "momentum": 0.9
        #     },
        "dropout_rates": None,      # a list of rates for each layer or None
        "conv_layer_specs": [       # activation can be "sigmoid", "tanh", "relu", "linear"
            {"filter_shape": (32, 1, 39, 9), "pool_shape": (1, 3), "activation": "relu"},
            {"filter_shape": (32, 32, 1, 8), "pool_shape": (1, 3), "activation": "relu"},
            ],
        "hidden_layer_specs": [
            {"units": 1024, "activation": "relu"},
            {"units": 1024, "activation": "linear"},
            ],
        # "test": [1, 3, 4]
        }

    results = sweep_nn_options(
        default_options_dict, eg_eval,
        # {"min_count": [3, 4, 5], "batch_size": [30, 40, 50]},
        n_hidden_units=[10, 20, 30],
        n_hidden_layers=[2, 3, 4],
        mode="serial"
        )
    for parameter_dict, result in results:
        print parameter_dict, result


if __name__ == "__main__":
    main()

