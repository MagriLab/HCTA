import argparse
from pathlib import Path

import tensorflow as tf

from hcta.experiment import ExperimentArgs
from hcta.experiment import main as run_experiment
from hcta.utils import config


def main(args):
    # load config file
    sweep_cfg = config.yaml2dict(args.sweep_config_path)

    # counter for the number of iterations (only for model related hyperparameters)
    # regularizations and learning rates are counted separately
    # load the base config file at the start
    base_cfg = config.yaml2dict(args.base_config_path)
    base_model_path = base_cfg["model_config"]["model_path"]
    for (lr_idx, lr) in enumerate(sweep_cfg["train_config"]["learning_rate"]):
        base_cfg["train_config"]["learning_rate"] = lr
        for (pi_idx, lambda_pi) in enumerate(sweep_cfg["train_config"]["lambda_pi"]):
            base_cfg["train_config"]["lambda_m"] = lambda_pi
            base_cfg["train_config"]["lambda_e"] = lambda_pi
            for (bs_idx, batch_size) in enumerate(
                sweep_cfg["data_config"]["batch_size"]
            ):
                base_cfg["data_config"]["batch_size"] = batch_size
                base_cfg["train_config"]["sampled_batch_size"] = batch_size
                for run_idx in range(sweep_cfg["N_runs"]):
                    # configure paths in the final loop
                    model_path = (
                        f'{sweep_cfg["name"]}/'
                        f'{base_cfg["model_config"]["model_type"]}'
                        f"_lr_{lr_idx+1}"
                        f"_pi_{pi_idx}"
                        f"_bs_{bs_idx}"
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
    parser = argparse.ArgumentParser(description="Sweep run")

    # arguments to define config paths for sweep run
    parser.add_argument(
        "--base-config-path", type=Path, default="src/configs/config.yml"
    )
    parser.add_argument(
        "--sweep-config-path", type=Path, default="src/configs/sweep_config_new.yml"
    )
    # arguments for weights and biases
    parser.add_argument("--wandb-entity", default=None, type=str)
    parser.add_argument("--wandb-project", default=None, type=str)
    parser.add_argument("--wandb-group", default=None, type=str)

    parsed_args = parser.parse_args()
    main(parsed_args)
