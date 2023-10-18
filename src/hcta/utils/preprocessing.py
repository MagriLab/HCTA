import h5py
import numpy as np
import scipy.io as spio
import tensorflow as tf
from sklearn.model_selection import train_test_split as interp_train_val_split
from sklearn.preprocessing import StandardScaler


def read_h5(path):
    """Read from simulation dictionary in a .h5 file

    Args:
        path: file path to data
    Returns:
        data_dictionary: dictionary that contains the items in the h5 file
    """
    data_dict = {}
    with h5py.File(path, "r") as hf:
        for k in hf.keys():
            data_dict[k] = hf.get(k)[()]
    return data_dict


def read_mat(path):
    """Read from simulation struct in a .mat file (generated in Matlab)
        If the data was already processed in matlab.
    Args:
        path: file path to data
    Returns:
        data_dictionary: dictionary that contains x,t,P,U on a grid
    """
    data_dict = spio.loadmat(path, squeeze_me=True)
    return data_dict


def _mat_check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    mat2dict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], spio.matlab.mat_struct):
            dict[key] = _mat2dict(dict[key])
    return dict


def _mat2dict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mat_struct):
            dict[strg] = _mat2dict(elem)
        else:
            dict[strg] = elem
    return dict


def load_mat(filename):
    """
    (this function should be called instead of direct scipy.io .loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects)

    https://stackoverflow.com/a/65195623

    Load the raw simulation data for processing
    """
    sim_mat = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    # sim.mat files return a struct called "Sim" at the highest level
    sim_dict = _mat_check_keys(sim_mat)["Sim"]
    return sim_dict


def sim2data_dict(sim_dict):
    """
    Extract data from the sim dictionary

    Shift x so that the inlet end is x = 0
    Transpose P,U,Q to match the dimensions in the other functions

    Args:
        sim_dict: dictionary that contains all simulation data and properties
                 generated from wave approach with G equation code
    Returns:
        data_dict: dictionary that contains x,t,P,U on a grid
    """
    data_dict = {
        "x": sim_dict["Measurement"]["Mic_Pos"] + sim_dict["Geom"]["Lu"],
        "t": sim_dict["t_mic"],
        "P": sim_dict["p_mic"].T,
        "U": sim_dict["u_mic"].T,
        "Q": sim_dict["Qs"].T,
    }
    return data_dict


def get_scales(sim_dict):
    """
    Returns the scales used to nondimensionalise data
    """
    # total length of the tube
    L = sim_dict["Geom"]["Lu"] + sim_dict["Geom"]["Lb"]

    # weighted average of speed of sound of mean flow in different regions
    c_avg = (
        sim_dict["Geom"]["Lu"] * sim_dict["Mean"]["c1"]
        + sim_dict["Geom"]["Lb"] * sim_dict["Mean"]["c2"]
    ) / L

    # weighted average of densities of mean flow in different regions
    rho_avg = (
        sim_dict["Geom"]["Lu"] * sim_dict["Mean"]["rho1"]
        + sim_dict["Geom"]["Lb"] * sim_dict["Mean"]["rho2"]
    ) / L

    x_scale = L
    t_scale = L / c_avg
    p_scale = rho_avg * c_avg**2
    u_scale = c_avg
    rho_scale = rho_avg
    q_scale = rho_avg * c_avg**3

    scales = {
        "x": x_scale,
        "t": t_scale,
        "p": p_scale,
        "u": u_scale,
        "rho": rho_scale,
        "q": q_scale,
    }
    return scales


def nondimensionalise(data_dict, scales):
    """
    Nondimensionalise the given data
    """
    data_dict["x"] = data_dict["x"] / scales["x"]
    data_dict["t"] = data_dict["t"] / scales["t"]
    data_dict["P"] = data_dict["P"] / scales["p"]
    data_dict["U"] = data_dict["U"] / scales["u"]
    data_dict["Q"] = data_dict["Q"] / scales["q"]
    return


def dimensionalise(data_dict, scales):
    """
    Nondimensionalise the given data
    """
    data_dict["x"] = data_dict["x"] * scales["x"]
    data_dict["t"] = data_dict["t"] * scales["t"]
    data_dict["P"] = data_dict["P"] * scales["p"]
    data_dict["U"] = data_dict["U"] * scales["u"]
    data_dict["Q"] = data_dict["Q"] * scales["q"]
    return


def discard_transient(data_dict, t_transient):
    """Discard the transient in a time series

    Args:
        data_dict: dictionary that contains x,t,P,U on a grid
        t_transient: transient time
    Returns:
        modified data_dict
    """
    dt = data_dict["t"][1] - data_dict["t"][0]
    N_0 = int(np.ceil(t_transient / dt))
    data_dict["t"] = data_dict["t"][N_0:] - data_dict["t"][N_0]
    data_dict["P"] = data_dict["P"][N_0:, :]
    data_dict["U"] = data_dict["U"][N_0:, :]
    data_dict["Q"] = data_dict["Q"][N_0:]
    return


def get_std(y):
    """Return the standard deviation of the data

    Args:
        y: data
    Returns:
        sigma: standard deviation of y over the whole domain
    """
    # Computes the standard deviation of the flattened array
    sigma = np.std(y)
    return sigma


def generate_noise(size, mean, sigma, seed):
    # @TODO: signal-to-noise ratio
    # check if seed produces same output every time
    """Generates Gaussian noise with given mean and standard deviation

    Args:
        size: shape of the noise matrix
        mean, sigma: mean and standard deviation of the Gaussian
        seed
    Returns:
        W: noise matrix with shape 'size'
    """
    rand = np.random.RandomState(seed)
    W = rand.normal(loc=mean, scale=sigma, size=size)
    return W


def set_sampling_time(data_dict, dt):
    """Change the sampling time of the given data. Can only be greater than the original."""
    dt_data = data_dict["t"][1] - data_dict["t"][0]
    t_step = int(
        np.ceil(dt / dt_data)
    )  # can only set the sampling time to be greater than the original one
    data_dict["t"] = data_dict["t"][0::t_step]
    data_dict["P"] = data_dict["P"][0::t_step, :]
    data_dict["U"] = data_dict["U"][0::t_step, :]
    data_dict["Q"] = data_dict["Q"][0::t_step]
    return


def remove_boundaries(data_dict, boundary):
    """If the given data contains the boundaries, remove them"""
    left_boundary = boundary[0]
    right_boundary = boundary[1]
    boundary_idx = np.logical_or(
        data_dict["x"] == left_boundary, data_dict["x"] == right_boundary
    )
    not_boundary_idx = np.logical_not(boundary_idx)
    data_dict["x"] = data_dict["x"][not_boundary_idx]
    data_dict["P"] = data_dict["P"][:, not_boundary_idx]
    data_dict["U"] = data_dict["U"][:, not_boundary_idx]
    return


def train_val_test_split(data_dict, t_train_len, t_val_len, t_test_len):
    # @TODO: this would be much better as a nested dictionary
    """Split the data in train, validation and test sets given the time lengths for each set

    Args:
        data_dict: dictionary that contains x,t,P,U on a grid
        t_train_len,t_val_len,t_test_len: time lengths of each dataset
    Returns:
        split_data_dict: dictionary that contains x,t,P,U split in train,val,test
    """
    dt = data_dict["t"][1] - data_dict["t"][0]

    # Split data into training, validation and test
    N_train = int(np.ceil(t_train_len / dt)) + 1
    N_val = int(np.ceil(t_val_len / dt)) + 1
    N_test = int(np.ceil(t_test_len / dt)) + 1

    t_train = data_dict["t"][:N_train]
    t_val = data_dict["t"][N_train : N_train + N_val]
    t_test = data_dict["t"][N_train + N_val : N_train + N_val + N_test]

    P_train = data_dict["P"][:N_train, :]
    P_val = data_dict["P"][N_train : N_train + N_val, :]
    P_test = data_dict["P"][N_train + N_val : N_train + N_val + N_test, :]

    U_train = data_dict["U"][:N_train, :]
    U_val = data_dict["U"][N_train : N_train + N_val, :]
    U_test = data_dict["U"][N_train + N_val : N_train + N_val + N_test, :]

    Q_train = data_dict["Q"][:N_train]
    Q_val = data_dict["Q"][N_train : N_train + N_val]
    Q_test = data_dict["Q"][N_train + N_val : N_train + N_val + N_test]

    split_data_dict = {
        "x_train": data_dict["x"],
        "x_val": data_dict["x"],
        "x_test": data_dict["x"],
        "t_train": t_train,
        "t_val": t_val,
        "t_test": t_test,
        "P_train": P_train,
        "P_val": P_val,
        "P_test": P_test,
        "U_train": U_train,
        "U_val": U_val,
        "U_test": U_test,
        "Q_train": Q_train,
        "Q_val": Q_val,
        "Q_test": Q_test,
    }
    return split_data_dict


def create_input_output(x, t, P, P_true, U, U_true):
    """Create input and output data from the data given on a grid
    'true' refers to noise-free data
    """
    [xx, tt] = np.meshgrid(x, t)
    x_data = xx.flatten()
    t_data = tt.flatten()
    P_data = P.flatten()
    U_data = U.flatten()
    P_true_data = P_true.flatten()
    U_true_data = U_true.flatten()
    input = np.hstack((x_data[:, None], t_data[:, None]))
    output = np.hstack((P_data[:, None], U_data[:, None]))
    output_true = np.hstack((P_true_data[:, None], U_true_data[:, None]))
    return input, output, output_true


# @todo: can't set interp_val_split to 0
def prepare_dataset(
    split_data_dict,
    split_data_dict_true,
    batch_size,
    standardise=False,
    my_scaler=None,
    interp_val_split=0,
    interp_val_split_seed=None,
):
    """Prepare the input and output data, convert to tf datasets

    Args:
        split_data_dict: dictionary that contains data split into train, val, test sets (used for training)
        split_data_dict_true: dictionary that contains the noise-free data (used for metrics)
        batch_size
        standardise: if true, data will be standardised
        scaler: provide a scaler if the data should be scaled with a specific scaler before training
    Returns:
        input_output_dict: dictionary that contains the input-output data
        train_dataset: training tf datasets fed to 'train' method
        val_interp_dataset, val_extrap_dataset: validation tf datasets fed to 'train' method
            We have to two validation datasets to assess interpolation and extrapolation
    """
    # get the full training set
    input_train_full, output_train_full, output_train_true_full = create_input_output(
        split_data_dict["x_train"],
        split_data_dict["t_train"],
        split_data_dict["P_train"],
        split_data_dict_true["P_train"],
        split_data_dict["U_train"],
        split_data_dict_true["U_train"],
    )

    # We consider two sets of validation,
    # one that can assess the interpolation capability, which should be within the training range

    # and one that can assess the extrapolation capability, which should be beyond the training range
    # split the train set into train and validation (interpolation) sets
    idx_train_full = np.arange(len(input_train_full))
    if interp_val_split == 0:
        input_train = input_train_full
        output_train = output_train_full
        output_train_true = output_train_true_full
        idx_train = idx_train_full
        input_val_interp = np.array([])
        output_val_interp = np.array([])
        output_val_interp_true = np.array([])
        idx_val_interp = np.array([])
    elif interp_val_split > 0:
        # store the indices in array to return the train and validation indices later
        (
            input_train,
            input_val_interp,
            output_train,
            output_val_interp,
            output_train_true,
            output_val_interp_true,
            idx_train,
            idx_val_interp,
        ) = interp_train_val_split(
            input_train_full,
            output_train_full,
            output_train_true_full,
            idx_train_full,
            test_size=interp_val_split,
            random_state=interp_val_split_seed,
        )
    else:
        raise ValueError("Train validation (interpolation) split should be >= 0.")

    # get the validation (extrapolation) set
    input_val_extrap, output_val_extrap, output_val_extrap_true = create_input_output(
        split_data_dict["x_val"],
        split_data_dict["t_val"],
        split_data_dict["P_val"],
        split_data_dict_true["P_val"],
        split_data_dict["U_val"],
        split_data_dict_true["U_val"],
    )

    # get the test set
    input_test, output_test, output_test_true = create_input_output(
        split_data_dict["x_test"],
        split_data_dict["t_test"],
        split_data_dict["P_test"],
        split_data_dict_true["P_test"],
        split_data_dict["U_test"],
        split_data_dict_true["U_test"],
    )

    # scale the data if necessary
    if standardise is True and my_scaler is None:
        scaler = StandardScaler()
        input_train = scaler.fit_transform(input_train)
        input_train_full = scaler.transform(input_train_full)
        input_val_interp = scaler.transform(input_val_interp)
        input_val_extrap = scaler.transform(input_val_extrap)
        input_test = scaler.transform(input_test)
    elif my_scaler is not None:
        input_train = my_scaler.transform(input_train)
        input_train_full = my_scaler.transform(input_train_full)
        input_val_interp = my_scaler.transform(input_val_interp)
        input_val_extrap = my_scaler.transform(input_val_extrap)
        input_test = my_scaler.transform(input_test)

    # typecast data to float32
    input_train = input_train.astype(np.float32)
    input_train_full = input_train_full.astype(np.float32)
    input_val_interp = input_val_interp.astype(np.float32)
    input_val_extrap = input_val_extrap.astype(np.float32)
    input_test = input_test.astype(np.float32)

    output_train = output_train.astype(np.float32)
    output_train_full = output_train_full.astype(np.float32)
    output_train_true = output_train_true.astype(np.float32)
    output_train_true_full = output_train_true_full.astype(np.float32)
    output_val_interp = output_val_interp.astype(np.float32)
    output_val_interp_true = output_val_interp_true.astype(np.float32)
    output_val_extrap = output_val_extrap.astype(np.float32)
    output_val_extrap_true = output_val_extrap_true.astype(np.float32)
    output_test = output_test.astype(np.float32)
    output_test_true = output_test.astype(np.float32)

    # prepare the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (input_train, output_train, output_train_true)
    )
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(
        batch_size
    )  # shuffle and divide in batches

    # prepare the validation (interpolation) dataset
    if interp_val_split > 0:
        val_interp_dataset = tf.data.Dataset.from_tensor_slices(
            (input_val_interp, output_val_interp, output_val_interp_true)
        )
        val_interp_dataset = val_interp_dataset.shuffle(buffer_size=1024).batch(
            batch_size
        )  # shuffle and divide in batches
    else:
        val_interp_dataset = None

    # prepare the validation (extrapolation) dataset
    val_extrap_dataset = tf.data.Dataset.from_tensor_slices(
        (input_val_extrap, output_val_extrap, output_val_extrap_true)
    )
    val_extrap_dataset = val_extrap_dataset.shuffle(buffer_size=1024).batch(
        batch_size
    )  # shuffle and divide in batches

    input_output_dict = {
        "input_train_full": input_train_full,
        "input_train": input_train,
        "input_val_interp": input_val_interp,
        "input_val_extrap": input_val_extrap,
        "input_test": input_test,
        "output_train_full": output_train_full,
        "output_train": output_train,
        "output_train_true_full": output_train_true_full,
        "output_train_true": output_train_true,
        "output_val_interp": output_val_interp,
        "output_val_interp_true": output_val_interp_true,
        "output_val_extrap": output_val_extrap,
        "output_val_extrap_true": output_val_extrap_true,
        "output_test": output_test,
        "output_test_true": output_test_true,
        "idx_train": idx_train,
        "idx_val_interp": idx_val_interp,
    }
    return input_output_dict, train_dataset, val_interp_dataset, val_extrap_dataset
