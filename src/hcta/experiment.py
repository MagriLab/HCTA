import argparse
import pickle
from pathlib import Path
from shutil import copyfile

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Activation, Lambda
from wandb.wandb_run import Run

import wandb

import numpy as np

import hcta.postprocessing as post
from .models.fnn import ForwardKinematicNN, ForwardRijkeNN
from .models.gnn import GalerkinKinematicNN, GalerkinRijkeNN
from .models.harmonics import Harmonics
from .rijke_galerkin.solver import Rijke
from .utils import config
from .utils import preprocessing as pp
from .utils import visualizations as vis


class ExperimentArgs:
    def __init__(
        self, config_path, wandb_entity=None, wandb_group=None, wandb_project=None
    ):
        self.config_path = config_path
        self.wandb_entity = wandb_entity
        self.wandb_group = wandb_group
        self.wandb_project = wandb_project


def load_data_kinematic(data_config: config.DataConfig):
    """Loads the kinematic flame model data from the given data path and preprocesses it.
        Discards the transient and nondimensionalises the data.
        Returns the preprocessed data and the nondimensionalised mean flow variables.
    Args:
        data_config: config object that contains the configurations for data loading and pre-processing
    """
    # load the simulation data from the mat file
    sim_dict = pp.load_mat(data_config.data_path)

    # extract the data used in training
    data_dict = pp.sim2data_dict(sim_dict)

    # discard the transient
    pp.discard_transient(
        data_dict, t_transient=sim_dict["t_transient"]
    )  # should save the transient time in the model, because t starts from t = 0

    # set sampling time
    pp.set_sampling_time(data_dict, data_config.dt)

    # nondimensionalise
    scales = pp.get_scales(sim_dict)
    pp.nondimensionalise(data_dict, scales)

    # remove the boundaries
    # after non-dimensionalising the boundaries are 0 and 1
    if not data_config.boundary:
        pp.remove_boundaries(data_dict, boundary=(0, 1))

    # save the non-dimensionalised mean-flow variables in a dictionary
    # would be nicer to have a kinematic flame object for this
    mean_flow_dict = {
        "A1": sim_dict["Geom"]["A1"],
        "A2": sim_dict["Geom"]["A2"],
        "x_f": sim_dict["Geom"]["Lu"] / scales["x"],
        "rho_up": sim_dict["Mean"]["rho1"] / scales["rho"],
        "rho_down": sim_dict["Mean"]["rho2"] / scales["rho"],
        "u_up": sim_dict["Mean"]["u1"] / scales["u"],
        "u_down": sim_dict["Mean"]["u2"] / scales["u"],
        "p_up": sim_dict["Mean"]["p1"] / scales["p"],
        "p_down": sim_dict["Mean"]["p2"] / scales["p"],
        "c_up": sim_dict["Mean"]["c1"] / scales["u"],
        "c_down": sim_dict["Mean"]["c2"] / scales["u"],
        "gamma": sim_dict["Mean"]["gamma"],
        "q_bar": sim_dict["Mean"]["Qbar"] / scales["q"],
    }
    return data_dict, mean_flow_dict


def load_data_rijke(
    data_config: config.DataConfig, model_config: config.ModelConfig = None
):
    """
    Loads the data rijke tube model from the given data path and preprocesses it.
    Discards the transient.
    Returns the preprocessed data and a Rijke object that contains the system parameters.
    Args:
        data_config: dictionary that contains the configurations for data loading and pre-processing
    """
    # load the data used in training
    data_dict = pp.read_h5(data_config.data_path)

    # create rijke object that contains the properties

    # For the Rijke system, we always create a Rijke model that has the same number of Galerkin modes
    # as the simulation. This object is only used for physics-informed training. This is a simplification,
    # so that the energy equation holds (heat release rate changes with the number of modes).
    if (model_config is not None) and (model_config.N_g != data_dict["N_g"]):
        print(
            f"Warning: Number of Galerkin modes passed in the config file is different from the simulation data. \n",
            f"Using the value in the simulation file to create the Rijke object. \n",
            f"Note this object is only used for the physics-informed training.",
        )

    rijke = Rijke(
        N_g=data_dict["N_g"],
        N_c=data_dict["N_c"],
        c_1=data_dict["c_1"],
        c_2=data_dict["c_2"],
        beta=data_dict["beta"],
        x_f=data_dict["x_f"],
        tau=data_dict["tau"],
        heat_law=data_dict["heat_law"].decode("utf-8"),
        damping=data_dict["damping"].decode("utf-8"),
    )

    # discard the transient
    pp.discard_transient(
        data_dict, t_transient=data_dict["t_transient"]
    )  # should save the transient time in the model, because t starts from t = 0

    # set sampling time
    pp.set_sampling_time(data_dict, data_config.dt)

    # remove the boundaries
    if not data_config.boundary:
        pp.remove_boundaries(data_dict, boundary=(0, 1))

    return data_dict, rijke


def load_data(
    data_config: config.DataConfig,
    model_config: config.ModelConfig = None,
    my_scaler=None,
):  # load and preprocess
    """
    Loads the data from the given data path and prepares it for training.
    Args:
        data_config: dictionary that contains the configurations for data loading and pre-processing
    Returns:
        mean_flow_dict: dictionary that contains the mean flow variables used while creating some of the models
        input_output_dict: dictionary that contains the input-output data
        train_dataset, val_dataset: training and validation tf datasets that should be fed to 'train' method
    """
    if data_config.data_type == "rijke":
        data_dict, rijke = load_data_rijke(data_config, model_config)
    elif data_config.data_type == "kinematic" or data_config.data_type == "experiment":
        data_dict, mean_flow_dict = load_data_kinematic(data_config)

    # add noise
    sigma_P = pp.get_std(data_dict["P"])
    W_P = pp.generate_noise(
        data_dict["P"].shape,
        mean=0.0,
        sigma=(data_config.noise.pressure_std / 100) * sigma_P,
        seed=data_config.noise.pressure_seed,
    )
    sigma_U = pp.get_std(data_dict["U"])
    W_U = pp.generate_noise(
        data_dict["U"].shape,
        mean=0.0,
        sigma=(data_config.noise.velocity_std / 100) * sigma_U,
        seed=data_config.noise.velocity_seed,
    )
    data_dict_true = data_dict.copy()  # store the true values for comparison
    data_dict["P"] = data_dict["P"] + W_P
    data_dict["U"] = data_dict["U"] + W_U

    if data_config.noise.pressure_std > 0:
        power_true_P = np.sum(data_dict_true["P"] ** 2)
        power_noise_P = np.sum(W_P**2)
        snr_P = 10 * np.log10(power_true_P / power_noise_P)
        print("SNR pressure:", snr_P)
    if data_config.noise.velocity_std > 0:
        power_true_U = np.sum(data_dict_true["U"] ** 2)
        power_noise_U = np.sum(W_U**2)
        snr_U = 10 * np.log10(power_true_U / power_noise_U)
        print("SNR velocity:", snr_U)

    # split data
    split_data_dict_true = pp.train_val_test_split(
        data_dict=data_dict_true,
        t_train_len=data_config.train_length,
        t_val_len=data_config.val_length,
        t_test_len=data_config.test_length,
    )
    split_data_dict = pp.train_val_test_split(
        data_dict=data_dict,
        t_train_len=data_config.train_length,
        t_val_len=data_config.val_length,
        t_test_len=data_config.test_length,
    )

    (
        input_output_dict,
        train_dataset,
        val_interp_dataset,
        val_extrap_dataset,
    ) = pp.prepare_dataset(
        split_data_dict=split_data_dict,
        split_data_dict_true=split_data_dict_true,
        batch_size=data_config.batch_size,
        standardise=data_config.standardise,
        my_scaler=my_scaler,
        interp_val_split=data_config.interp_val_split,
        interp_val_split_seed=data_config.interp_val_split_seed,
    )
    if data_config.data_type == "rijke":
        return (
            rijke,
            split_data_dict_true,
            split_data_dict,
            input_output_dict,
            train_dataset,
            val_interp_dataset,
            val_extrap_dataset,
        )
    elif data_config.data_type == "kinematic" or data_config.data_type == "experiment":
        return (
            mean_flow_dict,
            split_data_dict_true,
            split_data_dict,
            input_output_dict,
            train_dataset,
            val_interp_dataset,
            val_extrap_dataset,
        )


def create_model(model_config: config.ModelConfig, d_in=2, d_out=2):
    """
    Creates model given a model_dictionary
    Args:
        model_config: dictionary that contains the neural network parameters
    Returns:
        model: tf sequential model
    """
    if model_config.model_type == "fnn":
        d_in = 2
        d_out = 2
    elif model_config.model_type == "gnn":
        d_in = 1
        d_out = 2 * model_config.N_g
    else:
        print(
            "Model type not specified, using the default number of input and outputs."
        )
    if model_config.use_jump:
        d_out = d_out + 2

    if model_config.use_linear:
        d_out = d_out + 2

    # define model
    model = tf.keras.Sequential()
    # input layer
    model.add(tf.keras.layers.InputLayer(input_shape=d_in, dtype=tf.float32))

    # hidden layers
    for l in tf.range(model_config.N_layers):
        if model_config.activations[l] == "harmonics":
            model.add(Harmonics(model_config.N_neurons[l]))
        else:
            model.add(
                tf.keras.layers.Dense(
                    model_config.N_neurons[l],
                    kernel_initializer=model_config.initializers[l],
                    kernel_regularizer=model_config.regularizers[l],
                    dtype=tf.float32,
                )
            )

            # activation; sinx, relu, tanh...
            # if the activation is periodic, then set the hyperparameter a
            if model_config.activations[l].__name__ in ("sinx", "xSinx", "xSin2x"):
                model.add(
                    Activation(
                        Lambda(
                            lambda x, l: model_config.activations[l](x, model_config.a),
                            arguments={"l": l},
                        )
                    )
                )
            else:
                model.add(
                    Activation(
                        Lambda(
                            lambda x, l: model_config.activations[l](x),
                            arguments={"l": l},
                        )
                    )
                )

        if model_config.dropout_rate:
            model.add(
                tf.keras.layers.Dropout(
                    model_config.dropout_rate,
                    noise_shape=None,
                    seed=None,
                    dtype=tf.float32,
                )
            )

    # output layer
    model.add(
        tf.keras.layers.Dense(
            d_out,
            activation="linear",
            kernel_initializer=model_config.initializers[-1],
            kernel_regularizer=model_config.regularizers[-1],
            dtype=tf.float32,
        )
    )
    return model


def create_network(
    my_config: config.Config,
    model: tf.keras.Sequential,
    rijke: Rijke = None,
    dx: float = None,
    mean_flow: dict = None,
):
    """Creates a neural network object from the given model and parameters"""
    # check if the training is physics-informed
    # if it is then Rijke object shouldn't be none for 'rijke' or
    # mean flow dictionary shouldn't be none for 'kinematic'
    if (my_config.train_config.lambda_m != 0) or (my_config.train_config.lambda_e != 0):
        if my_config.data_config.data_type == "rijke" and not isinstance(rijke, Rijke):
            raise ValueError(
                "Need a Rijke object to train a physics-informed network on Rijke data."
            )
        elif my_config.data_config.data_type == "kinematic" and not isinstance(
            mean_flow, dict
        ):
            # @todo: would be nicer to have a meanflow object so it would also check the entries
            raise ValueError(
                "Need a mean flow dictionary to train a physics-informed network on kinematic model data."
            )
    else:
        # if the training is not physics-informed, then Rijke object should be switched to None
        # so that we can use the number of Galerkin modes specified in the config
        # because otherwise this will be overriden by the Rijke object
        rijke = None
        dx = None
        # don't want to make meanflow none because the Galerkin network depends
        # on the mean density values to create the acoustic modes
        # setting meanflow to none would require some more if conditions, which is not necessary

    # determine data type
    if my_config.data_config.data_type == "rijke":
        # choose type of network
        if my_config.model_config.model_type == "fnn":
            my_nn = ForwardRijkeNN(model=model, rijke=rijke, dx=dx)
        elif my_config.model_config.model_type == "gnn":
            my_nn = GalerkinRijkeNN(
                model=model, N_g=my_config.model_config.N_g, rijke=rijke
            )
    elif (
        my_config.data_config.data_type == "kinematic"
        or my_config.data_config.data_type == "experiment"
    ):
        # take a subset of the mean flow variables to pass into the network
        mean_flow_subset = {
            k: mean_flow[k]
            for k in (
                "x_f",
                "rho_up",
                "rho_down",
                "u_up",
                "u_down",
                "p_up",
                "p_down",
                "gamma",
            )
        }
        # choose type of network
        if my_config.model_config.model_type == "fnn":
            my_nn = ForwardKinematicNN(model=model, **mean_flow_subset)
        elif my_config.model_config.model_type == "gnn":
            my_nn = GalerkinKinematicNN(
                model=model,
                N_g=my_config.model_config.N_g,
                use_mean_flow=my_config.model_config.use_mean_flow,
                use_jump=my_config.model_config.use_jump,
                use_linear=my_config.model_config.use_linear,
                **mean_flow_subset,
            )
    return my_nn


def plot_P_U(pred_dict, split_data_dict, log_dir):
    # Plot pressure and velocity predictions obtained with the forward neural network (3d plots)
    dataset_name_list = ["train", "val", "test"]
    plot_path = log_dir / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    for dataset_name in dataset_name_list:
        fig = vis.surf(
            split_data_dict["x_" + dataset_name],
            split_data_dict["t_" + dataset_name],
            split_data_dict["P_" + dataset_name],
            pred_dict["P_" + dataset_name],
            "x",
            "t",
            "P " + dataset_name,
        )
        P_plot_name = "P_" + dataset_name + "_surf"
        plt.savefig(plot_path / P_plot_name)
        plt.close(fig)

        fig = vis.surf(
            split_data_dict["x_" + dataset_name],
            split_data_dict["t_" + dataset_name],
            split_data_dict["U_" + dataset_name],
            pred_dict["U_" + dataset_name],
            "x",
            "t",
            "U " + dataset_name,
        )
        U_plot_name = "U_" + dataset_name + "_surf"
        plt.savefig(plot_path / U_plot_name)
        plt.close(fig)


def main(args):
    """Runs the experiment.
    Args:
        args: Arguments provided at the command line
    """
    # load config file
    cfg = config.load_config(args.config_path)

    # save the config file
    copyfile(args.config_path, cfg.model_config.model_path / "config.yml")

    # start a weights and biases run to log the experiment
    # initialise weights and biases run
    if args.wandb_project is not None:
        cfg_dict = config.yaml2dict(args.config_path)
        my_wandb_run = wandb.init(
            config=cfg_dict,
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.wandb_group,
            reinit=True,
        )
    else:
        my_wandb_run = None

    # create neural network model
    model = create_model(cfg.model_config)

    # Learning rate schedule
    exp_decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=cfg.train_config.learning_rate,
        decay_steps=2000,
        decay_rate=0.9,
    )

    def lr_schedule(step):
        if cfg.train_config.learning_rate_schedule == "exponential_decay":
            lr = exp_decay_schedule(step)
        elif cfg.train_config.learning_rate_schedule == "constant":
            lr = cfg.train_config.learning_rate
        return lr

    # load the data
    if cfg.data_config.data_type == "rijke":
        (
            rijke,
            _,
            split_data_dict,
            _,
            train_dataset,
            val_interp_dataset,
            val_extrap_dataset,
        ) = load_data(cfg.data_config, cfg.model_config)
        mean_flow = None  # to prevent future references
    elif (
        cfg.data_config.data_type == "kinematic"
        or cfg.data_config.data_type == "experiment"
    ):
        (
            mean_flow,
            _,
            split_data_dict,
            _,
            train_dataset,
            val_interp_dataset,
            val_extrap_dataset,
        ) = load_data(cfg.data_config)
        rijke = None  # to prevent future references

    # create the neural network
    dx = split_data_dict["x_train"][1] - split_data_dict["x_train"][0]

    my_nn = create_network(
        my_config=cfg, model=model, rijke=rijke, dx=dx, mean_flow=mean_flow
    )

    # which domains to sample points to evaluate physics-informed loss
    sampled_domain = {
        "x": (0, 1),
        "t_train": (0, cfg.data_config.train_length),
        "t_val": (
            cfg.data_config.train_length,
            cfg.data_config.train_length + cfg.data_config.val_length,
        ),
    }
    with tf.device("CPU:0"):
        # train the model
        hist = my_nn.train(
            log_dir=cfg.model_config.model_path,
            train_dataset=train_dataset,
            val_interp_dataset=val_interp_dataset,
            val_extrap_dataset=val_extrap_dataset,
            epochs=cfg.train_config.epochs,
            save_epochs=cfg.train_config.save_epochs,
            print_epoch_mod=10,
            lr_schedule=lr_schedule,
            lambda_dd=cfg.train_config.lambda_dd,
            lambda_m=cfg.train_config.lambda_m,
            lambda_e=cfg.train_config.lambda_e,
            sampled_batch_size=cfg.train_config.sampled_batch_size,
            sampled_domain=sampled_domain,
            observe=cfg.data_config.observe,
        )

    # save history to pickle
    with open(cfg.model_config.model_path / "history.pickle", "wb") as handle:
        pickle.dump(hist, handle)

    # predict on the fine grid and plot the prediction
    split_data_dict_true, _, _, pred_dict, _ = post.make_prediction(
        cfg, epoch=cfg.train_config.epochs
    )
    plot_P_U(pred_dict, split_data_dict_true, cfg.model_config.model_path)

    # calculate the data-driven residual on the fine grid
    R_dd_true = post.calculate_datadriven_residual(split_data_dict_true, pred_dict)

    # log the history on weights and biases
    # if wanted, these can be logged in nn.train simultaneously during the training
    if isinstance(my_wandb_run, Run):
        for epoch in range(wandb.config.train_config["epochs"]):
            my_wandb_run.log(
                {
                    "train_loss": hist["train_loss"][epoch],
                    "val_interp_loss": hist["val_interp_loss"][epoch],
                    "val_extrap_loss": hist["val_extrap_loss"][epoch],
                    "train_metric": hist["train_metric"][epoch],
                    "train_metric_true": hist["train_metric_true"][epoch],
                    "val_interp_metric": hist["val_interp_metric"][epoch],
                    "val_interp_metric_true": hist["val_interp_metric_true"][epoch],
                    "val_extrap_metric": hist["val_extrap_metric"][epoch],
                    "val_extrap_metric_true": hist["val_extrap_metric_true"][epoch],
                }
            )

        my_wandb_run.log(
            {
                "train_fine_metric_true_final": R_dd_true["train"],
                "val_fine_metric_true_final": R_dd_true["val"],
                "test_fine_metric_true_final": R_dd_true["test"],
            }
        )

        my_wandb_run.finish()


if __name__ == "__main__":
    # read arguments from command line
    parser = argparse.ArgumentParser(description="Experiment run")

    # argument to define config path for experiment run
    parser.add_argument("--config-path", type=Path, default="src/configs/config.yml")
    # arguments for weights and biases
    parser.add_argument("--wandb-entity", default=None, type=str)
    parser.add_argument("--wandb-project", default=None, type=str)
    parser.add_argument("--wandb-group", default=None, type=str)

    parsed_args = parser.parse_args()
    main(parsed_args)
