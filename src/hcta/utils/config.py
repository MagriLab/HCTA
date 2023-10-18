import dataclasses
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import tensorflow as tf
import yaml

from ..models import periodic_activations


@dataclass()
class NoiseConfig:
    pressure_std: float  # percentage standard deviation of the Gaussian noise to add on pressure
    pressure_seed: int  # seed of the noise on pressure
    velocity_std: float  # percentage standard deviation of the Gaussian noise to add on velocity
    velocity_seed: int  # seed of the noise on velocity


@dataclass()
class DataConfig:
    data_type: Literal[
        "rijke", "kinematic", "experiment"
    ]  # which system the data comes from
    data_path: Path  # load the data from
    dt: float  # sampling time (dimensional)
    standardise: bool  # whether to standardise the data
    noise: NoiseConfig  # properties of the added noise
    train_length: float  # length of the time series used for training (nondimensional)
    val_length: float  # length of the time series used for validation (nondimensional)
    test_length: float  # length of the time series used for test (nondimensional)
    batch_size: int  # batch size
    interp_val_split: float = (
        0  # how much of the training data we allocate to validation of interpolation
    )
    interp_val_split_seed: int = None  # seed of the train and validation split
    observe: Literal["both", "p", "u"] = "both"
    boundary: bool = (
        True  # if true, the data contains the boundaries, if false they will be removed
    )

    def __post_init__(self):
        # make the data_path a Path object if it is not already
        if not isinstance(self.data_path, Path):
            self.data_path = Path(self.data_path)

        # value check
        # check if the given data type is allowed
        self._allowed_data_types = set(["rijke", "kinematic", "experiment"])
        if self.data_type not in self._allowed_data_types:
            raise ValueError(
                f"Data type {self.data_type} not allowed. Choose from: {self._allowed_data_types}."
            )
        # check if the given observable is allowed
        self._allowed_observed_vars = set(["both", "p", "u"])
        if self.observe not in self._allowed_observed_vars:
            raise ValueError(
                f"Observed variable {self.observe} not allowed. Choose from: {self._allowed_observed_vars}."
            )


@dataclass()
class ModelConfig:
    model_type: Literal["fnn", "gnn"]  # type of model
    model_path: Path  # save the model weights to
    N_g: int  # number of Galerkin modes (only valid for 'gnn')
    use_mean_flow: bool  # if True, mean density is used to generate the piece-wise Galerkin modes (only valid for 'gnn')
    use_jump: bool  # if True, jump modes are used for velocity (only valid for 'gnn')
    N_layers: int  # number of hidden layers, i.e., not including output layer
    N_neurons: list[int]  # number of neurons in each hidden layer
    activations: list[callable] or list[str]  # activations in each hidden layer
    a: float  # hyperparameter for sine activation
    regularizers: list[callable] or list[str]  # regularizers in each layer
    lambdas: list[float]  # regularization coefficient in each layer
    initializers: list[callable] or list[str]  # weight initializers in each layer
    dropout_rate: float  # drop out
    use_linear: bool = False

    def __post_init__(self):
        # make the model_path a Path object if it is not already
        if not isinstance(self.model_path, Path):
            self.model_path = Path(self.model_path)

        # create the model path
        self.model_path.mkdir(parents=True, exist_ok=True)

        # comment here
        if (
            any(isinstance(act, str) for act in self.activations)
            or any(isinstance(init, str) for init in self.initializers)
            or any(isinstance(reg, str) for reg in self.regularizers)
        ):
            read_model_config(self)

        # value checks
        # check if the given model type is allowed
        self._allowed_model_types = set(["fnn", "gnn"])
        if self.model_type not in self._allowed_model_types:
            raise ValueError(
                f"Model type {self.model_type} not allowed. Choose from: {self._allowed_model_types}."
            )

        # check if sin activation is used but the hyperparameter a is 0
        activations_set = set(self.activations)
        periodic_activations_set = set(
            [
                periodic_activations.sinx,
                periodic_activations.xSinx,
                periodic_activations.xSin2x,
            ]
        )
        if activations_set.intersection(periodic_activations_set) and self.a == 0:
            raise ValueError(
                "Periodic activation chosen but the hyperparameter a is 0. Provide new a."
            )


@dataclass()
class TrainConfig:
    learning_rate: float  # learning rate
    learning_rate_schedule: Literal[
        "constant", "exponential_decay"
    ]  # learning rate scheduler
    epochs: int  # number of epochs
    save_epochs: list[int]  # which epochs to save
    lambda_dd: float  # data-driven loss weight
    lambda_m: float  # momentum loss weight
    lambda_e: float  # energy loss weight
    sampled_batch_size: int = 0

    def __post_init__(self):
        # value check
        self._allowed_schedulers = set(["constant", "exponential_decay"])
        if self.learning_rate_schedule not in self._allowed_schedulers:
            raise ValueError(
                f"Learning rate scheduler {self.learning_rate_schedule} not allowed. Choose from: {self._allowed_schedulers}."
            )


@dataclass()
class Config:
    data_config: DataConfig
    model_config: ModelConfig
    train_config: TrainConfig


def read_model_config(model_config: ModelConfig):
    """
    Reads the model config into functions that can be passed into the create model
    """
    # create dictionaries that map the keys to functions
    initializers_dict = {
        "periodic_uniform": periodic_activations.periodic_uniform,
        "periodic_normal": periodic_activations.periodic_normal,
        "glorot_uniform": tf.keras.initializers.glorot_uniform,
        "glorot_normal": tf.keras.initializers.glorot_normal,
        "he_uniform": tf.keras.initializers.he_uniform,
        "he_normal": tf.keras.initializers.he_normal,
    }

    activations_dict = {
        "harmonics": "harmonics",
        "sin": periodic_activations.sinx,
        "xsin": periodic_activations.xSinx,
        "xsin2": periodic_activations.xSin2x,
        "relu": tf.keras.activations.relu,
        "tanh": tf.keras.activations.tanh,
    }

    bad_inits = []  # initialise a bad inits
    for i, init in enumerate(model_config.initializers):
        # check if initializer is in the dictionary
        if init in initializers_dict.keys():
            model_config.initializers[i] = initializers_dict[init]
        # if it is a string then it should be in the dictionary, otherwise it can't
        # be converted to a callable and passed into the model
        elif isinstance(init, str) and init not in initializers_dict.keys():
            bad_inits.extend([init])
    # raise an error including all the non-viable initializers
    if len(bad_inits) > 0:
        raise ValueError(
            f"Initializers {bad_inits} are not in the dictionary possible values."
        )

    bad_acts = []  # initialise a bad acts
    for i, act in enumerate(model_config.activations):
        # check if activation is in the dictionary
        if act in activations_dict.keys():
            model_config.activations[i] = activations_dict[act]
        # if it is a string then it should be in the dictionary, otherwise it can't
        # be converted to a callable and passed into the model
        elif isinstance(act, str) and init not in activations_dict.keys():
            bad_acts.extend([act])
    # raise an error including all the non-viable activations
    if len(bad_acts) > 0:
        raise ValueError(
            f"Activations {bad_acts} are not in the dictionary possible values."
        )

    bad_regs = []
    for i, reg in enumerate(model_config.regularizers):
        if reg == "L1":
            model_config.regularizers[i] = tf.keras.regularizers.L1(
                model_config.lambdas[i]
            )
        elif reg == "L2":
            model_config.regularizers[i] = tf.keras.regularizers.L2(
                model_config.lambdas[i]
            )
        elif reg == "None":
            model_config.regularizers[i] = None
        # if it is a string then it should be in the dictionary, otherwise it can't
        # be converted to a callable and passed into the model
        elif isinstance(reg, str):
            bad_regs.extend([reg])
    # raise an error including all the non-viable regularizations
    if len(bad_regs) > 0:
        raise ValueError(
            f"Regularizations {bad_regs} are not in the dictionary possible values."
        )
    return


def yaml2dict(config_path):
    """Load config dictionary from a config file."""
    with open(config_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


def dict2yaml(config_path, config_dict):
    """Save config dictionary as a config file."""
    with open(config_path, "w+") as ymlfile:
        yaml.dump(config_dict, ymlfile)
    return


def dict2dataclass(klass, d):
    """Make a dataclass from nested dictionary
    klass: dataclass
    d: dictionary
    https://stackoverflow.com/a/54769644
    """
    try:
        fieldtypes = {f.name: f.type for f in dataclasses.fields(klass)}
        return klass(**{f: dict2dataclass(fieldtypes[f], d[f]) for f in d})
    except ValueError as ve:
        print(repr(ve))
        raise
    except:
        return d  # Not a dataclass field


def load_config(config_path):
    """Load a yaml config file and create a config object."""
    cfg = yaml2dict(config_path)
    # create config object
    cfg_obj = dict2dataclass(Config, cfg)
    return cfg_obj
