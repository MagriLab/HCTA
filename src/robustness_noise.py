import argparse
import os
import sys
from pathlib import Path

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)
import tensorflow as tf

from hcta.experiment import ExperimentArgs
from hcta.experiment import main as run_experiment
from hcta.utils import config


def main(args):
    # load config file
    rbst_cfg = config.yaml2dict(args.robustness_config_path)

    # load the base config file at the start
    base_cfg = config.yaml2dict(args.base_config_path)
    base_model_path = base_cfg["model_config"]["model_path"]
    for noise_std in rbst_cfg["data_config"]["noise_std"]:
        base_cfg["data_config"]["noise"]["pressure_std"] = noise_std
        base_cfg["data_config"]["noise"]["velocity_std"] = noise_std
        for (lambda_pi_idx, lambda_pi) in enumerate(
            rbst_cfg["train_config"]["lambda_pi"]
        ):
            base_cfg["train_config"]["lambda_m"] = lambda_pi
            base_cfg["train_config"]["lambda_e"] = lambda_pi
            for run_idx in range(rbst_cfg["N_runs"]):
                base_cfg["data_config"]["noise"]["pressure_seed"] = 2 * run_idx + 1
                base_cfg["data_config"]["noise"]["velocity_seed"] = 2 * run_idx + 2
                # configure paths in the final loop
                model_path = (
                    f"noise_{noise_std}/"
                    f'{base_cfg["model_config"]["model_type"]}'
                    f"_pi_{lambda_pi_idx}"
                    f"_run_{run_idx+1}"
                )
                base_cfg["model_config"]["model_path"] = (
                    base_model_path + "/" + model_path
                )
                temp_config_path = Path("src/configs/temp_config.yml")
                config.dict2yaml(temp_config_path, base_cfg)
                print("Config saved.")

                # run the experiment with the saved config path
                print("Running experiment.")
                tf.keras.backend.clear_session()
                experiment_args = ExperimentArgs(
                    config_path=temp_config_path,
                    wandb_entity=args.wandb_entity,
                    wandb_group=args.wandb_group,
                    wandb_project=args.wandb_project,
                )
                run_experiment(experiment_args)
                print("Experiment finished.")


if __name__ == "__main__":
    # read arguments from command line
    parser = argparse.ArgumentParser(
        description="Robustness with respect to noise level"
    )

    # arguments to define config paths for sweep run
    parser.add_argument(
        "--base-config-path", type=Path, default="src/configs/config.yml"
    )
    parser.add_argument(
        "--robustness-config-path",
        type=Path,
        default="src/configs/robustness_noise_config.yml",
    )
    # arguments for weights and biases
    parser.add_argument("--wandb-entity", default=None, type=str)
    parser.add_argument("--wandb-project", default=None, type=str)
    parser.add_argument("--wandb-group", default=None, type=str)

    parsed_args = parser.parse_args()
    main(parsed_args)
