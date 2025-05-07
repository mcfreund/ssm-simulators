import logging
import pickle
import yaml
from copy import deepcopy
import warnings
from pathlib import Path

import typer

import ssms

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = typer.Typer()


def try_gen_folder(
    folder: str | Path | None = None, allow_abs_path_folder_generation: bool = True
) -> None:
    """Function to generate a folder from a string. If the folder already exists, it will not be generated.

    Arguments
    ---------
        folder (str):
            The folder string to generate.
        allow_abs_path_folder_generation (bool):
            If True, the folder string is treated as an absolute path.
            If False, the folder string is treated as a relative path.
    """
    if not folder:
        raise ValueError("Folder path cannot be None or empty.")

    folder_path = Path(folder)

    # Check if the path is absolute and if absolute path generation is allowed
    if folder_path.is_absolute() and not allow_abs_path_folder_generation:
        warnings.warn(
            "Absolute folder path provided, but allow_abs_path_folder_generation is False. "
            "No folders will be generated."
        )
        return

    try:
        # Create the folder and any necessary parent directories
        folder_path.mkdir(parents=True, exist_ok=True)
        logging.info("Folder %s created or already exists.", folder_path)
    except Exception as e:
        logging.error("Error creating folder '%s': %s", folder, e)


def make_data_generator_configs(
    model="ddm",
    generator_approach="lan",
    data_generator_arg_dict=None,
    model_config_arg_dict=None,
    save_name=None,
    save_folder="",
):
    # Load copy of the respective model's config dict from ssms
    model_config = deepcopy(ssms.config.model_config[model.split("_deadline")[0]])

    # Load copy of the respective data_generator_config dicts
    data_config = deepcopy(ssms.config.data_generator_config[generator_approach])
    data_config["model"] = model

    for key, val in data_generator_arg_dict.items():
        data_config[key] = val

    for key, val in model_config_arg_dict.items():
        model_config[key] = val

    config_dict = {"model_config": model_config, "data_config": data_config}

    if save_name is not None:
        if len(save_folder) > 0:
            if save_folder[-1] == "/":
                pass
            else:
                save_folder = save_folder + "/"

        # Create save_folder if not already there
        try_gen_folder(folder=save_folder, allow_abs_path_folder_generation=True)

        # Dump pickle file
        with open(save_folder + save_name, "wb") as f:
            pickle.dump(config_dict, f)

        print("Saved to: ")
        print(save_folder + save_name)

    return {
        "config_dict": config_dict,
        "config_file_name": None if save_name is None else save_folder + save_name,
    }


def get_data_generator_config(yaml_config_path=None, base_path=None):
    with open(yaml_config_path, "rb") as f:
        basic_config = yaml.safe_load(f)

    training_data_folder = (
        base_path
        + "data/training_data/"
        + basic_config["GENERATOR_APPROACH"]
        + "/training_data_n_samples_"
        + str(basic_config["N_SAMPLES"])
        + "_dt_"
        + str(basic_config["DELTA_T"])
        + "/"
        + str(basic_config["MODEL"])
    )

    data_generator_arg_dict = {
        "output_folder": training_data_folder,
        "model": basic_config["MODEL"],
        "n_samples": basic_config["N_SAMPLES"],
        "n_parameter_sets": basic_config["N_PARAMETER_SETS"],
        "delta_t": basic_config["DELTA_T"],
        "n_training_samples_by_parameter_set": basic_config[
            "N_TRAINING_SAMPLES_BY_PARAMETER_SET"
        ],
        "n_subruns": basic_config["N_SUBRUNS"],
        "cpn_only": True if (basic_config["GENERATOR_APPROACH"] == "cpn") else False,
    }

    model_config_arg_dict = {}

    config_dict = make_data_generator_configs(
        model=basic_config["MODEL"],
        generator_approach=basic_config["GENERATOR_APPROACH"],
        data_generator_arg_dict=data_generator_arg_dict,
        model_config_arg_dict=model_config_arg_dict,
        save_name=None,
        save_folder=None,
    )
    return config_dict


@app.command()
def main(
    config_path: Path = typer.Option(None, help="Path to the YAML configuration file."),
    output: Path = typer.Option(None, help="Path to the output directory."),
    yaml_file: Path = typer.Option(
        None, help="Path to a YAML file containing arguments."
    ),
):
    """
    Generate data using the specified configuration.
    """
    if yaml_file:
        # Load arguments from the YAML file
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)
        config_path = Path(config["config_path"])
        output = Path(config["output"])

    if not config_path or not output:
        typer.echo("Both --config-path and --output must be provided.")
        raise typer.Exit(code=1)

    # Casting config_path to str for now
    # TODO: Fix this in the future
    config_path = str(config_path)
    output = str(output)

    config_dict = get_data_generator_config(
        yaml_config_path=config_path, base_path=output
    )["config_dict"]

    logging.info("Printing config specs:")
    logging.info("GENERATOR CONFIG")
    logging.info(config_dict["data_config"])

    logging.info("MODEL CONFIG")
    logging.info(config_dict["model_config"])

    # Make the generator
    logging.info("Now generating data")
    my_dataset_generator = ssms.dataset_generators.lan_mlp.data_generator(
        generator_config=config_dict["data_config"],
        model_config=config_dict["model_config"],
    )
    if "cpn_only" in config_dict["data_config"]:
        if config_dict["data_config"]["cpn_only"]:
            my_dataset_generator.generate_data_training_uniform(
                save=True, cpn_only=True
            )
        else:
            my_dataset_generator.generate_data_training_uniform(
                save=True, cpn_only=False
            )
    logging.info("Data generation finished")


if __name__ == "__main__":
    app()
