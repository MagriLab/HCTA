import argparse
from pathlib import Path

import tensorflow as tf

from hcta.experiment import ExperimentArgs
from hcta.experiment import main as run_experiment
from hcta.utils import config


def set_N_layers(N_layers, config_dict):
    """Sets the number of layers in the config dictionary.
    Args:
        N_layers: number of hidden layers
        config_dict: config dictionary
    """
    config_dict["model_config"]["N_layers"] = N_layers
    return


def set_N_neurons(N_neurons, N_layers, config_dict):
    """Sets the number of neurons in the config dictionary such that,
    there are same number of neurons in each hidden layer.
    Args:
        N_neuron: number of neurons
        N_layers: number of hidden layers
        config_dict: config dictionary
    """
    config_dict["model_config"]["N_neurons"] = N_layers * [N_neurons]
    return


def set_activations(activations, N_layers, config_dict):
    """Sets the activations and related parameters in the config dictionary such that,
    if only one activation is given, there is the same activation in each hidden layer,
    if two activations are given e.g. sin-relu, then sin activation is in the first hidden layer,
    and the other one is in the rest of the hidden layers.
    Weight initialization and standardization are set according to the activation function.
    Args:
        activations: activation function given as string
        N_layers: number of hidden layers
        config_dict: config dictionary
    Returns:
        run_flag: whether to run an experiment or not, used in order to skip running two activations string,
            e.g. sin-relu and 1 layer case and not to crash the whole sweep
    """
    if activations in set(["relu", "tanh", "sin", "xsin", "xsin2", "harmonics"]):
        # set the activation in all layers
        config_dict["model_config"]["activations"] = N_layers * [activations]
        # set the weight initialisation and standardisation
        # add one to the N_layers to get the total number of layers including the output layer
        if activations == "relu":
            config_dict["model_config"]["initializers"] = (N_layers + 1) * [
                "he_uniform"
            ]
            config_dict["data_config"]["standardise"] = True
        elif activations == "tanh":
            config_dict["model_config"]["initializers"] = (N_layers + 1) * [
                "glorot_uniform"
            ]
            config_dict["data_config"]["standardise"] = True
        else:
            config_dict["model_config"]["initializers"] = (N_layers + 1) * [
                "periodic_uniform"
            ]
            config_dict["data_config"]["standardise"] = False
        run_flag = True
    elif activations in set(["sin-relu", "sin-tanh"]):
        if N_layers >= 2:
            # set the activation of the first layer
            activations_list = ["sin"]
            # set the standardization
            config_dict["data_config"]["standardise"] = False
            # set the weight initialization of the first layer
            initializers_list = ["periodic_uniform"]
            if activations == "sin-relu":
                # set the activation of the remaining layers
                activations_list.extend((N_layers - 1) * ["relu"])
                config_dict["model_config"]["activations"] = activations_list
                # set the weight initialization of the remaining layers
                initializers_list.extend(N_layers * ["he_uniform"])
                config_dict["model_config"]["initializers"] = initializers_list
            else:
                # set the activation of the remaining layers
                activations_list.extend((N_layers - 1) * ["tanh"])
                config_dict["model_config"]["activations"] = activations_list
                # set the weight initialization of the remaining layers
                initializers_list.extend(N_layers * ["glorot_uniform"])
                config_dict["model_config"]["initializers"] = initializers_list
            run_flag = True
        else:
            run_flag = False
    else:
        run_flag = False
        print(f"Activation {activations} not defined.")
    return run_flag


def set_regularizations(lambdas, N_layers, config_dict):
    """Sets the regularizations in the config dictionary such that,
    the same regularization (type and coefficient) is applied in each layer.
    Args:
        lambdas: regularization coefficient
        N_layers: number of hidden layers
        config_dict: config dictionary
    """
    # set the regularization coefficients
    # add one to the N_layers to get the total number of layers including the output layer
    config_dict["model_config"]["lambdas"] = (N_layers + 1) * [lambdas]
    # extend the regularization type to all layers
    config_dict["model_config"]["regularizers"] = (N_layers + 1) * [
        config_dict["model_config"]["regularizers"][0]
    ]
    return


def set_learning_rate(learning_rate, config_dict):
    """Sets the learning rate in the config dictionary.
    Args:
        learning_rate: learning rate
        config_dict: config dictionary
    """
    config_dict["train_config"]["learning_rate"] = learning_rate
    return


def main(args):
    # load config file
    sweep_cfg = config.yaml2dict(args.sweep_config_path)

    # @todo2: other things that can be varied, noise (std), number of galerkin modes for gnn, lambdas for pi loss

    # counter for the number of iterations (only for model related hyperparameters)
    # regularizations and learning rates are counted separately
    n_iter = 1
    # load the base config file at the start
    base_cfg = config.yaml2dict(args.base_config_path)
    base_model_path = base_cfg["model_config"]["model_path"]
    for N_layers in sweep_cfg["model_config"]["N_layers"]:
        set_N_layers(N_layers, config_dict=base_cfg)
        for N_neurons in sweep_cfg["model_config"]["N_neurons"]:
            set_N_neurons(N_neurons, N_layers, config_dict=base_cfg)
            for acts in sweep_cfg["model_config"]["activations"]:
                n_iter += 1
                run_flag = set_activations(acts, N_layers, config_dict=base_cfg)
                for lmbds_idx, lmbds in enumerate(sweep_cfg["model_config"]["lambdas"]):
                    set_regularizations(lmbds, N_layers, config_dict=base_cfg)
                    for lr_idx, lr in enumerate(
                        sweep_cfg["train_config"]["learning_rate"]
                    ):
                        set_learning_rate(lr, config_dict=base_cfg)
                        if run_flag:
                            for run_idx in range(sweep_cfg["N_runs"]):
                                # configure paths in the final loop
                                model_path = (
                                    f'{sweep_cfg["name"]}/'
                                    f'{base_cfg["model_config"]["model_type"]}'
                                    f"_{n_iter}"
                                    f"_reg_{lmbds_idx+1}"
                                    f"_lr_{lr_idx+1}"
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
                        else:
                            print("Skipping experiment.")


if __name__ == "__main__":
    # read arguments from command line
    parser = argparse.ArgumentParser(description="Sweep run")

    # arguments to define config paths for sweep run
    parser.add_argument(
        "--base-config-path", type=Path, default="src/configs/config.yml"
    )
    parser.add_argument(
        "--sweep-config-path", type=Path, default="src/configs/sweep_1_config.yml"
    )
    # arguments for weights and biases
    parser.add_argument("--wandb-entity", default=None, type=str)
    parser.add_argument("--wandb-project", default=None, type=str)
    parser.add_argument("--wandb-group", default=None, type=str)

    parsed_args = parser.parse_args()
    main(parsed_args)
