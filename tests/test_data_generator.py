from copy import deepcopy

import numpy as np
import pytest

from ssms.dataset_generators.lan_mlp import data_generator
from ssms.config import model_config, data_generator_config

gen_config = data_generator_config["lan"]
# Specify number of parameter sets to simulate
gen_config["n_parameter_sets"] = 100
# Specify how many samples a simulation run should entail
gen_config["n_samples"] = 1000

# angle
expected_shapes = {
    "gamma_drift": {
        "cpn_data": (100, 7),
        "cpn_labels": (100,),
        "cpn_no_omission_data": (100, 7),
        "cpn_no_omission_labels": (100,),
        "opn_data": (100, 7),
        "opn_labels": (100, 1),
        "gonogo_data": (100, 7),
        "gonogo_labels": (100, 1),
        "thetas": (100, 7),
        "lan_data": (100000, 9),
        "lan_labels": (100000,),
        "binned_128": (100, 128, 2),
        "binned_256": (100, 256, 2),
    },
    "ddm_sdv": {
        "cpn_data": (100, 5),
        "cpn_labels": (100,),
        "cpn_no_omission_data": (100, 5),
        "cpn_no_omission_labels": (100,),
        "opn_data": (100, 5),
        "opn_labels": (100, 1),
        "gonogo_data": (100, 5),
        "gonogo_labels": (100, 1),
        "thetas": (100, 5),
        "lan_data": (100000, 7),
        "lan_labels": (100000,),
        "binned_128": (100, 128, 2),
        "binned_256": (100, 256, 2),
    },
    "ddm_rayleight": {
        "cpn_data": (100, 4),
        "cpn_labels": (100,),
        "cpn_no_omission_data": (100, 4),
        "cpn_no_omission_labels": (100,),
        "opn_data": (100, 4),
        "opn_labels": (100, 1),
        "gonogo_data": (100, 4),
        "gonogo_labels": (100, 1),
        "thetas": (100, 4),
        "lan_data": (100000, 6),
        "lan_labels": (100000,),
        "binned_128": (100, 128, 2),
        "binned_256": (100, 256, 2),
    },
    "ddm_truncnormt": {
        "cpn_data": (100, 5),
        "cpn_labels": (100,),
        "cpn_no_omission_data": (100, 5),
        "cpn_no_omission_labels": (100,),
        "opn_data": (100, 5),
        "opn_labels": (100, 1),
        "gonogo_data": (100, 5),
        "gonogo_labels": (100, 1),
        "thetas": (100, 5),
        "lan_data": (100000, 7),
        "lan_labels": (100000,),
        "binned_128": (100, 128, 2),
        "binned_256": (100, 256, 2),
    },
    "ddm_st": {
        "cpn_data": (100, 5),
        "cpn_labels": (100,),
        "cpn_no_omission_data": (100, 5),
        "cpn_no_omission_labels": (100,),
        "opn_data": (100, 5),
        "opn_labels": (100, 1),
        "gonogo_data": (100, 5),
        "gonogo_labels": (100, 1),
        "thetas": (100, 5),
        "lan_data": (100000, 7),
        "lan_labels": (100000,),
        "binned_128": (100, 128, 2),
        "binned_256": (100, 256, 2),
    },
    "full_ddm_rv": {
        "cpn_data": (100, 7),
        "cpn_labels": (100,),
        "cpn_no_omission_data": (100, 7),
        "cpn_no_omission_labels": (100,),
        "opn_data": (100, 7),
        "opn_labels": (100, 1),
        "gonogo_data": (100, 7),
        "gonogo_labels": (100, 1),
        "thetas": (100, 7),
        "lan_data": (100000, 9),
        "lan_labels": (100000,),
        "binned_128": (100, 128, 2),
        "binned_256": (100, 256, 2),
    },
    "ddm_legacy": {
        "cpn_data": (100, 4),
        "cpn_labels": (100,),
        "cpn_no_omission_data": (100, 4),
        "cpn_no_omission_labels": (100,),
        "opn_data": (100, 4),
        "opn_labels": (100, 1),
        "gonogo_data": (100, 4),
        "gonogo_labels": (100, 1),
        "thetas": (100, 4),
        "lan_data": (100000, 6),
        "lan_labels": (100000,),
        "binned_128": (100, 128, 2),
        "binned_256": (100, 256, 2),
    },
    "angle": {
        "cpn_data": (100, 5),
        "cpn_labels": (100,),
        "cpn_no_omission_data": (100, 5),
        "cpn_no_omission_labels": (100,),
        "opn_data": (100, 5),
        "opn_labels": (100, 1),
        "gonogo_data": (100, 5),
        "gonogo_labels": (100, 1),
        "thetas": (100, 5),
        "lan_data": (100000, 7),
        "lan_labels": (100000,),
        "binned_128": (100, 128, 2),
        "binned_256": (100, 256, 2),
    },
    "ddm": {
        "cpn_data": (100, 4),
        "cpn_labels": (100,),
        "cpn_no_omission_data": (100, 4),
        "cpn_no_omission_labels": (100,),
        "opn_data": (100, 4),
        "opn_labels": (100, 1),
        "gonogo_data": (100, 4),
        "gonogo_labels": (100, 1),
        "thetas": (100, 4),
        "lan_data": (100000, 6),
        "lan_labels": (100000,),
        "binned_128": (100, 128, 2),
        "binned_256": (100, 256, 2),
    },
    "weibull": {
        "cpn_data": (100, 6),
        "cpn_labels": (100,),
        "cpn_no_omission_data": (100, 6),
        "cpn_no_omission_labels": (100,),
        "opn_data": (100, 6),
        "opn_labels": (100, 1),
        "gonogo_data": (100, 6),
        "gonogo_labels": (100, 1),
        "thetas": (100, 6),
        "lan_data": (100000, 8),
        "lan_labels": (100000,),
        "binned_128": (100, 128, 2),
        "binned_256": (100, 256, 2),
    },
    "levy": {
        "cpn_data": (100, 5),
        "cpn_labels": (100,),
        "cpn_no_omission_data": (100, 5),
        "cpn_no_omission_labels": (100,),
        "opn_data": (100, 5),
        "opn_labels": (100, 1),
        "gonogo_data": (100, 5),
        "gonogo_labels": (100, 1),
        "thetas": (100, 5),
        "lan_data": (100000, 7),
        "lan_labels": (100000,),
        "binned_128": (100, 128, 2),
        "binned_256": (100, 256, 2),
    },
    "levy_angle": {
        "cpn_data": (100, 6),
        "cpn_labels": (100,),
        "cpn_no_omission_data": (100, 6),
        "cpn_no_omission_labels": (100,),
        "opn_data": (100, 6),
        "opn_labels": (100, 1),
        "gonogo_data": (100, 6),
        "gonogo_labels": (100, 1),
        "thetas": (100, 6),
        "lan_data": (100000, 8),
        "lan_labels": (100000,),
        "binned_128": (100, 128, 2),
        "binned_256": (100, 256, 2),
    },
    "full_ddm": {
        "cpn_data": (100, 7),
        "cpn_labels": (100,),
        "cpn_no_omission_data": (100, 7),
        "cpn_no_omission_labels": (100,),
        "opn_data": (100, 7),
        "opn_labels": (100, 1),
        "gonogo_data": (100, 7),
        "gonogo_labels": (100, 1),
        "thetas": (100, 7),
        "lan_data": (100000, 9),
        "lan_labels": (100000,),
        "binned_128": (100, 128, 2),
        "binned_256": (100, 256, 2),
    },
}

expected_constrained_param_space = {
    "gamma_drift": {
        "v": (-3.0, 3.0),
        "a": (0.3, 3.0),
        "z": (0.1, 0.9),
        "t": (0.001, 2.0),
        "shape": (2.0, 10.0),
        "scale": (0.01, 1.0),
        "c": (-3.0, 3.0),
    },
    "ddm_sdv": {
        "v": (-3.0, 3.0),
        "a": (0.3, 2.5),
        "z": (0.1, 0.9),
        "t": (0.001, 2.0),
        "sv": (0.001, 2.5),
    },
    "ddm_rayleight": {
        "v": (-3.0, 3.0),
        "a": (0.3, 2.5),
        "z": (0.3, 0.7),
        "st": (0.001, 1.0),
    },
    "ddm_truncnormt": {
        "v": (-3.0, 3.0),
        "a": (0.3, 2.5),
        "z": (0.3, 0.7),
        "mt": (0.05, 2.25),
        "st": (0.001, 0.5),
    },
    "ddm_st": {
        "v": (-3.0, 3.0),
        "a": (0.3, 2.5),
        "z": (0.3, 0.7),
        "t": (0.25, 2.25),
        "st": (0.001, 0.25),
    },
    "full_ddm_rv": {
        "v": (-3.0, 3.0),
        "a": (0.3, 2.5),
        "z": (0.3, 0.7),
        "t": (0.25, 2.25),
        "sz": (0.001, 0.2),
        "sv": (0.001, 2.0),
        "st": (0.001, "t"),
    },
    "levy_angle": {
        "v": (-3.0, 3.0),
        "a": (0.3, 3.0),
        "z": (0.1, 0.9),
        "alpha": (1.0, 2.0),
        "t": (0.001, 2),
        "theta": (-0.1, 1.3),
    },
    "angle": {
        "a": (0.3, 3.0),
        "t": (0.001, 2.0),
        "theta": (-0.1, 1.3),
        "v": (-3.0, 3.0),
        "z": (0.1, 0.9),
    },
    "ddm": {"v": (-3.0, 3.0), "a": (0.3, 2.5), "z": (0.1, 0.9), "t": (0.0, 2.0)},
    "ddm_legacy": {"v": (-3.0, 3.0), "a": (0.3, 2.5), "z": (0.1, 0.9), "t": (0.0, 2.0)},
    "weibull": {
        "v": (-2.5, 2.5),
        "a": (0.3, 2.5),
        "z": (0.2, 0.8),
        "t": (0.001, 2.0),
        "alpha": (0.31, 4.99),
        "beta": (0.31, 6.99),
    },
    "levy": {
        "v": (-3.0, 3.0),
        "a": (0.3, 3.0),
        "z": (0.1, 0.9),
        "alpha": (1.0, 2.0),
        "t": (0.001, 2),
    },
    "full_ddm": {
        "v": (-3.0, 3.0),
        "a": (0.3, 2.5),
        "z": (0.3, 0.7),
        "t": (0.25, 2.25),
        "sz": (0.001, 0.2),
        "sv": (0.001, 2.0),
        "st": (0.001, 0.25),
    },
}


@pytest.mark.parametrize("model_name", list(model_config.keys())[25:])
def test_model_config(model_name):
    # Take an example config for a given model
    model_conf = model_config[model_name]

    assert type(model_conf["simulator"]).__name__ == "cython_function_or_method"

    assert callable(model_conf["simulator"])
    assert callable(model_conf["boundary"])


# @pytest
@pytest.mark.parametrize("model_name", model_config.keys())
def test_data_generator(tmp_path, model_name):
    # Initialize the generator config (for MLP LANs)

    generator_config = deepcopy(gen_config)
    # Specify generative model (one from the list of included models mentioned above)

    generator_config["dgp_list"] = model_name

    # set output folder
    generator_config["output_folder"] = str(tmp_path)

    # Now let's define our corresponding `model_config`.
    model_conf = model_config[model_name]

    with pytest.raises(ValueError):
        data_generator(generator_config=generator_config, model_config=None)

    with pytest.raises(ValueError):
        data_generator(generator_config=None, model_config=model_conf)

    my_dataset_generator = data_generator(
        generator_config=generator_config, model_config=model_conf
    )
    training_data = my_dataset_generator.generate_data_training_uniform(save=True)

    new_data_file = list(tmp_path.iterdir())[0]
    assert new_data_file.exists()
    assert new_data_file.suffix == ".dill"

    # Because randomly generated arrays may differ across OS and versions of Python,
    # even when setting a random seed, we check for array shape
    td_array_shapes = {
        k: v.shape for k, v in training_data.items() if isinstance(v, np.ndarray)
    }

    shapes = {model_name: td_array_shapes}
    space = {model_name: training_data["model_config"]["constrained_param_space"]}
    assert td_array_shapes == expected_shapes[model_name]

    assert list(training_data.keys()) == [
        "cpn_data",
        "cpn_labels",
        "cpn_no_omission_data",
        "cpn_no_omission_labels",
        "opn_data",
        "opn_labels",
        "gonogo_data",
        "gonogo_labels",
        "thetas",
        "lan_data",
        "lan_labels",
        "binned_128",
        "binned_256",
        "generator_config",
        "model_config",
    ]

    assert (
        training_data["model_config"]["constrained_param_space"]
        == expected_constrained_param_space[model_name]
    )

    del training_data["model_config"]["constrained_param_space"]
    sfp = "simulator_fixed_params"
    if (
        sfp in training_data["model_config"] and model_name in model_config
    ):  # can't compare these
        del training_data["model_config"][sfp]
        del model_config[model_name][sfp]
    assert training_data["model_config"] == model_config[model_name]
