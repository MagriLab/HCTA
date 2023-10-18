from pathlib import Path

import numpy as np
import tensorflow as tf
from numpy import pi
from sklearn.preprocessing import StandardScaler

import hcta.experiment as exp
from .utils import preprocessing as pp


def get_weight_path(model_path, epoch):
    if epoch is not None:
        weight_path = model_path / "models" / f"model_epochs_{epoch}"
    else:
        weight_path = model_path / "models" / "best_model"
    return weight_path


def load_network(my_config, epoch=None, rijke=None, dx=None, mean_flow=None):
    """Create and load a network given configurations and path of the weights"""
    # create neural network model
    model = exp.create_model(my_config.model_config)

    # create network
    my_nn = exp.create_network(
        my_config=my_config, model=model, rijke=rijke, dx=dx, mean_flow=mean_flow
    )

    # load the trained weights from the chosen epoch
    # write the path the load the weights from
    weight_path = get_weight_path(my_config.model_config.model_path, epoch)

    my_nn.model.load_weights(weight_path)
    return my_nn


def get_first_layer_weights(my_config, epoch):
    """Returns the weights of the network in the first layer.
    These correspond to the angular frequencies of the activations for periodic networks.
    For sine networks, the saved weights are multiplied with the hyperparameter 'a'
    For a Galerkin neural network with a trainable frequency (harmonics) layer,
    the trained dominant angular frequency is returned.
    """
    # create neural network model
    model = exp.create_model(my_config.model_config)

    # load the trained weights from the chosen epoch
    # write the path the load the weights from
    weight_path = get_weight_path(my_config.model_config.model_path, epoch)
    model.load_weights(weight_path)

    # get the trained angular frequency
    # this is the only trainable variable in the first layer
    weights = model.get_weights()[0]

    # multiply with the hyperparameter a for the sine networks
    if my_config.model_config.activations[0] != "harmonics":
        if my_config.model_config.model_type == "fnn":
            weights = weights[1]  # take only the weights that multiply time
        elif my_config.model_config.model_type == "gnn":
            weights = weights[0]
        weights = np.sort(weights)
        if my_config.model_config.activations[0].__name__ in (
            "sinx",
            "xSinx",
            "xSin2x",
        ):
            weights = my_config.model_config.a * weights
            # for these activations weights are multiplied with the hyperparameter a

    print("Trained angular frequency(ies) = ", weights)
    print("Period(s) = ", 2 * pi / weights)
    return weights


def load_prediction_data(my_config):
    """Loads the data used for prediction
    Scales it if necessary
    """
    # load the original data used for training
    if my_config.data_config.data_type == "rijke":
        (
            rijke,
            split_data_dict_true,
            split_data_dict,
            input_output_dict,
            _,
            _,
            _,
        ) = exp.load_data(my_config.data_config, my_config.model_config)
        mean_flow = None
    elif (
        my_config.data_config.data_type == "kinematic"
        or my_config.data_config.data_type == "experiment"
    ):
        (
            mean_flow,
            split_data_dict_true,
            split_data_dict,
            input_output_dict,
            _,
            _,
            _,
        ) = exp.load_data(my_config.data_config)
        rijke = None

    # return the indices of the data that has been used for training
    idx_train = input_output_dict["idx_train"]

    # the test data (or truth) can be different, e.g. on a different grid, the path is specified below
    if my_config.data_config.data_type == "kinematic":
        my_stem = my_config.data_config.data_path.stem
        try:
            my_sys_str = my_stem[my_stem.index("Sys") : my_stem.index("_Sensor")]
        except:
            my_sys_str = my_stem[my_stem.index("Sys") :]
        my_new_path = f"data/Sim_Wave_G_eqn_{my_sys_str}.mat"
        my_config.data_config.data_path = Path(my_new_path)
    elif my_config.data_config.data_type == "experiment":
        my_config.data_config.data_path = Path("data/Sim_Experiment_4.mat")
    elif my_config.data_config.data_type == "rijke":
        new_stem = my_config.data_config.data_path.stem + "_fine"
        my_config.data_config.data_path = my_config.data_config.data_path.with_stem(
            new_stem
        )
        my_config.data_config.boundary = True

    # if the training data has been standardised, then the test data should be too with the same scaling
    if my_config.data_config.standardise:
        input_train, _, _ = pp.create_input_output(
            split_data_dict["x_train"],
            split_data_dict["t_train"],
            split_data_dict["P_train"],
            split_data_dict_true["P_train"],
            split_data_dict["U_train"],
            split_data_dict_true["U_train"],
        )
        scaler = StandardScaler()
        scaler.fit(input_train)
        # load the test data which will serve as the truth
        _, split_data_dict_true, _, input_output_dict, _, _, _ = exp.load_data(
            my_config.data_config, my_scaler=scaler
        )
    else:
        _, split_data_dict_true, _, input_output_dict, _, _, _ = exp.load_data(
            my_config.data_config
        )

    return (
        split_data_dict_true,
        split_data_dict,
        idx_train,
        input_output_dict,
        rijke,
        mean_flow,
    )


def make_prediction(my_config, epoch):
    """Makes prediction on the loaded data"""
    # load prediction data
    (
        split_data_dict_true,
        split_data_dict,
        idx_train,
        input_output_dict,
        rijke,
        mean_flow,
    ) = load_prediction_data(my_config)

    # spacing in x domain, only required by fnn for the evaluation of physics informed loss
    dx = split_data_dict["x_train"][1] - split_data_dict["x_train"][0]

    # load the neural network with the given configurations
    my_nn = load_network(
        my_config=my_config, epoch=epoch, rijke=rijke, dx=dx, mean_flow=mean_flow
    )
    # predict on the given test set
    pred_dict = my_nn.predict_P_U(input_output_dict, split_data_dict_true)

    return split_data_dict_true, split_data_dict, idx_train, pred_dict, my_nn


def calculate_datadriven_residual(split_data_dict_true, pred_dict):
    """Calculate the data-driven residual on the true data"""
    dataset_name_list = ["train", "val", "test"]
    R_dd_true = {"train": [], "val": [], "test": []}
    for dataset_name in dataset_name_list:
        p_true = split_data_dict_true["P_" + dataset_name]
        p_pred = pred_dict["P_" + dataset_name]
        u_true = split_data_dict_true["U_" + dataset_name]
        u_pred = pred_dict["U_" + dataset_name]
        R_dd_true[dataset_name] = (
            1
            / 2
            * (
                np.mean(np.square(p_true - p_pred))
                + np.mean(np.square(u_true - u_pred))
            )
        )
    return R_dd_true


def calculate_physics_residual(my_nn, x, t):
    [xx, tt] = np.meshgrid(x, t)
    x_data = xx.flatten()
    t_data = tt.flatten()
    my_input = np.hstack((x_data[:, None], t_data[:, None]))
    my_input = tf.constant(my_input, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(my_input)
        my_output_pred = my_nn.predict(my_input, training=False)
    jac_pred = tape.batch_jacobian(my_output_pred, my_input)
    dpdx = jac_pred[:, 0, 0]
    dpdt = jac_pred[:, 0, 1]
    dudx = jac_pred[:, 1, 0]
    dudt = jac_pred[:, 1, 1]

    dpdx = dpdx.numpy()
    dpdt = dpdt.numpy()
    dudx = dudx.numpy()
    dudt = dudt.numpy()

    dpdx = dpdx.reshape(len(t), len(x))
    dpdt = dpdt.reshape(len(t), len(x))
    dudx = dudx.reshape(len(t), len(x))
    dudt = dudt.reshape(len(t), len(x))

    # check if data obeys the momentum law
    # rho*du/dt+dp/dx = 0
    up_idx = np.where(x <= my_nn.x_f)[0]
    down_idx = np.where(x > my_nn.x_f)[0]

    R_momentum_up = (
        dudt[:, up_idx]
        + my_nn.u_up * dudx[:, up_idx]
        + 1 / my_nn.rho_up * dpdx[:, up_idx]
    )
    R_momentum_down = (
        dudt[:, down_idx]
        + my_nn.u_down * dudx[:, down_idx]
        + 1 / my_nn.rho_down * dpdx[:, down_idx]
    )
    R_momentum = np.hstack((R_momentum_up, R_momentum_down))
    R_momentum_mse = np.mean(np.square(R_momentum))

    R_energy_up = (
        dpdt[:, up_idx]
        + my_nn.u_up * dpdx[:, up_idx]
        + my_nn.gamma * my_nn.p_up * dudx[:, up_idx]
    )
    R_energy_down = (
        dpdt[:, down_idx]
        + my_nn.u_down * dpdx[:, down_idx]
        + my_nn.gamma * my_nn.p_down * dudx[:, down_idx]
    )
    R_energy = np.hstack((R_energy_up, R_energy_down))
    R_energy_mse = np.mean(np.square(R_energy))

    # LOW ORDER
    # R_momentum_up = dudt[:,up_idx]+dpdx[:,up_idx]
    # R_momentum_down = dudt[:,down_idx]+dpdx[:,down_idx]
    # R_momentum = np.hstack((R_momentum_up,R_momentum_down))
    # R_momentum_mse = np.mean(np.square(R_momentum))
    #
    # R_energy_up = dpdt[:,up_idx]+dudx[:,up_idx]
    # R_energy_down = dpdt[:,down_idx]+dudx[:,down_idx]
    # R_energy = np.hstack((R_energy_up,R_energy_down))
    # R_energy_mse = np.mean(np.square(R_energy))
    return R_momentum_mse, R_energy_mse


def calculate_physics_residual_rijke(my_nn, x, t):
    [xx, tt] = np.meshgrid(x, t)
    x_data = xx.flatten()
    t_data = tt.flatten()
    my_input = np.hstack((x_data[:, None], t_data[:, None]))
    my_input = tf.constant(my_input, dtype=tf.float32)

    input_size = my_input.shape[0]

    with tf.GradientTape() as tape:
        tape.watch(my_input)
        my_output_pred = my_nn.predict(my_input, training=False)
    jac_pred = tape.batch_jacobian(my_output_pred, my_input)
    dpdx = jac_pred[:, 0, 0]
    dpdt = jac_pred[:, 0, 1]
    dudx = jac_pred[:, 1, 0]
    dudt = jac_pred[:, 1, 1]

    dpdx = dpdx.numpy()
    dpdt = dpdt.numpy()
    dudx = dudx.numpy()
    dudt = dudt.numpy()

    dpdx = dpdx.reshape(len(t), len(x))
    dpdt = dpdt.reshape(len(t), len(x))
    dudx = dudx.reshape(len(t), len(x))
    dudt = dudt.reshape(len(t), len(x))

    # Momentum equation
    # EVALUATE RESIDUAL FROM MOMENTUM EQUATION
    momentum_eqn = dudt + dpdx
    pi_momentum_loss = tf.reduce_mean(tf.square(momentum_eqn))

    # Energy equation
    # prepare input at t = t-tau
    input_tau = my_input[:, 1] - my_nn.rjk.tau
    input_tau = input_tau[:, None]
    # prepare input at x = x_f
    input_f_tau = tf.concat(
        [my_nn.rjk.x_f * tf.ones((input_size, 1)), input_tau], axis=1
    )
    # predict u(x = x_f, t = t-tau) using the model
    u_f_tau = my_nn.predict(input_f_tau, training=False)[:, 1]

    # compute heat release rate
    if my_nn.rjk.heat_law == "kings":
        q_dot = my_nn.rjk.beta * (tf.sqrt(tf.abs(1 + u_f_tau)) - 1)
    elif my_nn.rjk.heat_law == "sigmoid":
        q_dot = my_nn.rjk.beta / (1 + tf.exp(-u_f_tau))

    # heat_release = \sum_{j = 1}^N_g 2*\dot{q}*sin(j\pi x_f)*sin(j\pi x)
    sinjpix = tf.sin(tf.tensordot(my_input[:, 0], my_nn.jpi, axes=0))
    sum_jx = tf.tensordot(sinjpix, my_nn.sinjpixf, axes=1)
    heat_release = 2 * q_dot * sum_jx

    if my_nn.rjk.damping == "constant":
        # Damping term
        damping = my_nn.rjk.c_1 * my_output_pred[:, 0]
    elif my_nn.rjk.damping == "modal":
        # Damping
        x_grid = tf.linspace(0, 1, 3 * my_nn.rjk.N_g + 1)
        x_grid = tf.cast(x_grid, dtype=tf.float32)
        t_grid = my_input[:, 1]

        # create input data to evaluate on the whole x grid at each t in the batch
        input_grid = tf.meshgrid(x_grid, t_grid)
        input_x_grid = tf.reshape(input_grid[0], [-1])
        input_t_grid = tf.reshape(input_grid[1], [-1])
        input_x_grid = input_x_grid[:, None]
        input_t_grid = input_t_grid[:, None]
        input_grid_flat = tf.concat([input_x_grid, input_t_grid], axis=1)

        # predict pressure
        P_grid_flat = my_nn.predict(input_grid_flat, training=False)[:, 0]
        P_grid_real = tf.reshape(P_grid_flat, [len(t_grid), len(x_grid)])
        # create the pressure over x = [0,2) so that when we take the FFT
        # we can have pi as the first Fourier frequency (resolution)
        P_grid_real = tf.concat(
            [P_grid_real, -tf.reverse(P_grid_real[:, 1:-1], axis=[1])], axis=1
        )
        # the indices in the reverse are from 1 because we don't want to include 2
        # to -1 because we don't want to repeat 1
        P_grid_im = tf.zeros(P_grid_real.shape)
        P_grid_complex = tf.complex(P_grid_real, P_grid_im)  # convert to complex

        # transform pressure to frequency domain via fft
        P_grid_fft = tf.signal.fft(P_grid_complex)

        # find the damping modes
        zeta_im = tf.zeros(my_nn.zeta.shape, dtype=tf.float32)
        zeta = tf.complex(my_nn.zeta, zeta_im)

        P_grid_fft_pos = P_grid_fft[
            :, 1 : my_nn.rjk.N_g + 1
        ]  # take only the positive frequencies and only up to the N_g

        # apply damping
        # convolution in spatial domain is multiplication in the frequency domain
        conv1 = zeta * P_grid_fft_pos
        # find the reverse for the negative frequencies
        conv2 = tf.math.conj(tf.reverse(conv1, axis=[1]))
        conv = tf.concat(
            (
                conv1,
                conv2,
            ),
            axis=1,
        )

        # now we will do the inverse fft
        # get the frequencies (positive and negative)
        jpi_full = tf.concat([my_nn.jpi, tf.reverse(-my_nn.jpi, axis=[0])], axis=0)

        # multiply with the inputs
        jpi_full_x_im = tf.tensordot(
            my_input[:, 0], jpi_full, axes=0
        )  # this is the inside of the exponential
        jpi_full_x_real = tf.zeros(jpi_full_x_im.shape, dtype=tf.float32)
        jpi_full_x = tf.complex(jpi_full_x_real, jpi_full_x_im)

        # take the exponential
        exp_im_jpix = tf.exp(jpi_full_x)

        # multiply the exponential term with the damping values in frequency domain
        conv_exp = conv * exp_im_jpix

        # sum to get the real damping value
        damping = tf.reduce_sum(conv_exp, axis=1)

        # divide by the total number of frequencies
        damping = 1 / (6 * my_nn.rjk.N_g) * damping
        # N_g times 2 because positive and negative freq,
        # times 3 because we initially divided [0,1) into 3*N_g
        # the fft was performed on a total of 6*N_g points

        damping = tf.math.real(damping)

    # EVALUATE RESIDUAL FROM ENERGY EQUATION
    damping = damping.numpy()
    heat_release = heat_release.numpy()

    damping = damping.reshape(len(t), len(x))
    heat_release = heat_release.reshape(len(t), len(x))

    energy_eqn = dpdt + dudx + damping - heat_release
    pi_energy_loss = tf.reduce_mean(tf.square(energy_eqn))

    return momentum_eqn, pi_momentum_loss, energy_eqn, pi_energy_loss


def calculate_physics_residual_rijke_galerkin(my_nn, t):
    t = t[:, None]
    t = tf.constant(t, dtype=tf.float32)

    # use gradient tape for automatic differentiation of the network
    with tf.GradientTape() as pi_tape:
        pi_tape.watch(t)
        y_pred = my_nn.model(t, training=False)
    y_dot_pred = pi_tape.batch_jacobian(y_pred, t)

    # @codereview: abstracting this into a function
    mu_pred = y_pred[:, 0 : my_nn.N_g]
    eta_pred = y_pred[:, my_nn.N_g : 2 * my_nn.N_g]
    mu_dot_pred = y_dot_pred[:, 0 : my_nn.N_g, 0]  # @codereview: squeeze over dimension
    eta_dot_pred = y_dot_pred[:, my_nn.N_g : 2 * my_nn.N_g, 0]

    # Momentum equation
    # EVALUATE RESIDUAL FROM MOMENTUM EQUATION
    momentum_eqn = eta_dot_pred - my_nn.jpi * mu_pred
    pi_momentum_loss = tf.reduce_mean(tf.square(momentum_eqn))

    # Energy equation
    # predict time-delayed input for the batch points
    y_tau_pred = my_nn.model(t - my_nn.rjk.tau, training=False)
    eta_tau_pred = y_tau_pred[:, my_nn.N_g : 2 * my_nn.N_g]
    # compute time-delayed flame velocity
    u_f_tau = tf.tensordot(eta_tau_pred, my_nn.cosjpixf, axes=1)

    # compute heat release rate, @codereview: else statement raise some error, enums
    if my_nn.rjk.heat_law == "kings":
        q_dot = my_nn.rjk.beta * (tf.sqrt(tf.abs(1 + u_f_tau)) - 1)
    elif my_nn.rjk.heat_law == "sigmoid":
        q_dot = my_nn.rjk.beta / (1 + tf.exp(-u_f_tau))

    heat_release = 2 * tf.tensordot(q_dot, my_nn.sinjpixf, axes=0)

    # EVALUATE RESIDUAL FROM MOMENTUM EQUATION
    energy_eqn = (
        mu_dot_pred + my_nn.jpi * eta_pred + my_nn.zeta * mu_pred + heat_release
    )
    pi_energy_loss = tf.reduce_mean(tf.square(energy_eqn))
    return momentum_eqn, pi_momentum_loss, energy_eqn, pi_energy_loss
