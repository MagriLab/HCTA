from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

import hcta.experiment as exp
import hcta.postprocessing as post
from hcta.utils import config
from hcta.utils import preprocessing as pp
from hcta.utils import signals
from hcta.utils.config import DataConfig


def make_prediction(cfg, model_epoch, rijke, mean_flow, input, data_dict):
    my_nn = post.load_network(cfg, epoch=model_epoch, rijke=rijke, mean_flow=mean_flow)

    # if the training data was standardised then the input should be transformed too
    if cfg.data_config.standardise:
        (
            _,
            split_data_dict_true,
            split_data_dict,
            _,
            _,
            _,
            _,
        ) = exp.load_data(cfg.data_config)
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
        input = scaler.transform(input)

    # predict on the given input
    # return the predictions as dictionary
    output_pred = my_nn.predict(input, training=False).numpy()
    pred_dict = {}
    pred_dict["P"] = output_pred[:, 0].reshape(len(data_dict["t"]), len(data_dict["x"]))
    pred_dict["U"] = output_pred[:, 1].reshape(len(data_dict["t"]), len(data_dict["x"]))
    return pred_dict


def plot_weights(
    ax,
    weights,
    omega_harmonics,
    bg_color,
    model_color,
    inset_x_lim,
    harmonic_freq,
    dominant_freq,
    ylabel="$W^{(1)}_1$",
):
    # Plot the weights
    ax.scatter(range(len(weights)), weights, s=200, color=model_color, zorder=2)
    for omega_harmonic in omega_harmonics:
        ax.axhline(
            y=omega_harmonic,
            color="grey",
            linestyle="--",
            alpha=0.7,
            linewidth=5.5,
            zorder=1,
        )
    ax.set_xlabel("Neuron")
    ax.set_ylabel(ylabel)
    ax.set_yticks(omega_harmonics)
    ax.set_facecolor(bg_color)
    inset_ax = ax.inset_axes([0.5, 0.05, 0.45, 0.45])
    inset_ax.scatter(range(len(weights)), weights, s=200, color=model_color, zorder=2)
    for omega_harmonic in omega_harmonics:
        inset_ax.axhline(
            y=omega_harmonic,
            color="grey",
            linestyle="--",
            alpha=0.7,
            linewidth=5.5,
            zorder=1,
        )
    inset_ax.set_ylim([dominant_freq - 0.05, dominant_freq + 0.05])
    inset_ax.set_xlim(inset_x_lim)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([dominant_freq], fontsize=10)
    inset_ax.set_facecolor(bg_color)


def plot_long_term(ax, t, y, y_pred, train_length, model_color):
    # Plot the long term prediction
    rect = patches.Rectangle(
        [0, np.min(y) - 0.15 * np.max(y)],
        width=train_length,
        height=1.3 * np.max(y) - np.min(y),
        facecolor="red",
    )
    ax.add_patch(rect)
    ax.plot(0, 0, color="grey", linewidth=3.5)
    ax.plot(0, 0, color="darkorange", linewidth=3.5)
    ax.plot(t, y, color="grey", linewidth=1)
    ax.plot(t, y_pred, color="darkorange", linewidth=0.5, linestyle="-")
    ax.set_ylim([np.min(y) - 0.15 * np.max(y), 1.15 * np.max(y)])
    ax.set_xlabel("$t$")
    ax.set_ylabel("$p'(x,t)$")
    ax.legend(
        ["Train", "True", "Prediction"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.3),
        ncol=3,
        columnspacing=0.65,
        handletextpad=0.25,
        handlelength=1.5,
    )


def plot_psd(
    ax,
    omega,
    psd,
    psd_pred,
    bg_color,
    model_color,
    inset_x_lim,
    inset_y_lim,
    harmonic_freq,
    dominant_freq,
):
    linewidth = 3.5
    # Plot the power spectral density
    ax.plot(omega, psd, color="silver", linewidth=3.5 * linewidth)
    ax.plot(omega, psd_pred, color=model_color, linewidth=linewidth, linestyle="-")
    ax.set_xlim([0, 20])
    ax.set_xticks(harmonic_freq * np.arange(0, 7))
    ax.set_xlabel("$\omega$")
    ax.set_ylabel("$PSD(p'(x,t))$")
    ax.set_facecolor(bg_color)
    ax.grid()
    ax.legend(
        ["True", "Prediction"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.3),
        ncol=3,
        columnspacing=0.65,
        handletextpad=0.25,
        handlelength=1.5,
    )
    inset_ax = ax.inset_axes([0.5, 0.45, 0.45, 0.45])
    inset_ax.plot(omega, psd, color="silver", linewidth=3.5 * linewidth)
    inset_ax.plot(
        omega, psd_pred, color=model_color, linewidth=linewidth, linestyle="-"
    )
    inset_ax.set_yticks([])
    inset_ax.set_ylim(inset_y_lim)
    inset_ax.set_xlim(inset_x_lim)
    inset_ax.set_xticks([dominant_freq], fontsize=10)
    inset_ax.grid()
    inset_ax.set_facecolor(bg_color)


def main():
    # load true data
    data_config = DataConfig(
        data_type="rijke",
        data_path="data/rijke_kings_beta_5_7_tau_0_2_long.h5",
        standardise=False,
        dt=0.001,
        noise=None,
        train_length=None,
        val_length=None,
        test_length=None,
        batch_size=None,
    )

    if data_config.data_type == "rijke":
        data_dict, rijke = exp.load_data_rijke(data_config)
        mean_flow = None
    elif data_config.data_type == "kinematic":
        data_dict, mean_flow = exp.load_data_kinematic(data_config)
        rijke = None

    # choose which x location to plot the spectrum of
    # extract the true pressure time series
    dx = data_dict["x"][1] - data_dict["x"][0]
    x_plot = 0.25
    x_plot_idx = int(np.round(x_plot / dx))
    my_p = data_dict["P"][:, x_plot_idx]

    # find period
    dt = data_dict["t"][1] - data_dict["t"][0]
    T_period = signals.period(my_p, dt)
    data_omega = 2 * np.pi / T_period
    print("True omega = ", data_omega)
    print("True period = ", T_period)
    # take the maximum number of periods
    # the real period isn't an exact multiple of the sampling time
    # therefore, the signal doesn't repeat itself at exact integer indices
    # so calculating the number of time steps in each period
    # does not work in order to cut the signal at the maximum number of periods
    # that's why we will cut between peaks, which is a more reliable option
    # though still not exact
    (start_pk_idx, end_pk_idx) = signals.periodic_signal_peaks(my_p, T_period - 0.1)
    my_p_periodic = my_p[start_pk_idx:end_pk_idx]

    # find psd
    omega, psd = signals.power_spectral_density(my_p_periodic, dt)
    # to get the dominan frequency from the psd
    psd_max_idx = np.argsort(psd)  # sort the psd
    data_omega_psd = np.min(
        omega[psd_max_idx[-5:]]
    )  # look at the 5 freqs with max psd and get the min frequency
    data_dominant_psd = omega[psd_max_idx[-1]]
    if np.abs(data_dominant_psd - data_omega_psd) < (data_omega_psd - 0.1):
        # need to add this condition because sometimes the max psds are around the dominant freq
        data_omega_psd = data_dominant_psd
    data_omega_harmonics = data_omega_psd * np.arange(-20, 20, 1)
    data_period_psd = 2 * np.pi / data_omega_psd
    print("True omega = ", data_omega_psd)
    print("True period = ", data_period_psd)
    # create input for the model
    my_input, _, _ = pp.create_input_output(
        data_dict["x"],
        data_dict["t"],
        data_dict["P"],
        data_dict["P"],
        data_dict["U"],
        data_dict["U"],
    )
    my_input = my_input.astype(np.float32)

    # load networks
    if data_config.data_type == "rijke":
        model_paths = [
            Path("figure_data/figure_6_7/model_1"),
            Path("figure_data/figure_6_7/model_2"),
            Path("figure_data/figure_6_7/model_3"),
        ]

        inset_x_lim = [[21.5, 25.5], [7.5, 10.5], [0.5, 1.5]]
        model_colors = ["black", "black", "black"]
        model_names = ["FNN", "GalNN", "P-GalNN"]

    harmonic_freq = data_omega_psd
    dominant_freq = data_dominant_psd
    model_epochs = [None, None, None]
    fontsize = 55
    # bg_color = [170 / 256, 190 / 256, 210 / 256]
    # bg_color = adjust_lightness(bg_color, 1.1)
    bg_color = "white"

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = fontsize

    fig_w = plt.figure(constrained_layout=True, figsize=(38, 12))
    subfigs_w = fig_w.subfigures(1, 3)

    fig_long = plt.figure(constrained_layout=True, figsize=(38, 27))
    subfigs_long = fig_long.subfigures(3, 2)

    titles = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    ylabels = ["$aW^{(1)}_2$", "$aW^{(1)}$", "$W^{(1)}$"]
    for model_idx, (model_path, model_epoch, model_color, model_name) in enumerate(
        zip(model_paths, model_epochs, model_colors, model_names)
    ):
        # load config
        config_path = model_path / "config.yml"
        cfg = config.load_config(config_path)

        # predict
        pred_dict = make_prediction(
            cfg, model_epoch, rijke, mean_flow, my_input, data_dict
        )

        # extract the predicted pressure time series
        my_p_pred = pred_dict["P"][:, x_plot_idx]

        # cut the predicted pressure time series
        my_p_pred_periodic = my_p_pred[start_pk_idx:end_pk_idx]

        # find psd of the prediction
        _, psd_pred = signals.power_spectral_density(my_p_pred_periodic, dt)

        # Plot the weights
        weights = post.get_first_layer_weights(cfg, model_epoch)

        if model_name == "P-GalNN":
            weights = weights * np.arange(1, cfg.model_config.N_neurons[0])
        ax_w = subfigs_w[model_idx].add_subplot(1, 1, 1)
        ax_w.set_title(titles[model_idx], loc="left")
        plot_weights(
            ax_w,
            weights,
            data_omega_harmonics,
            bg_color,
            model_color,
            inset_x_lim[model_idx],
            harmonic_freq,
            dominant_freq,
            ylabels[model_idx],
        )
        if model_name == "P-GalNN":
            ax_w.set_ylim([0, 40])
        else:
            ax_w.set_ylim([-20, 20])
        # Plot the long term prediction
        train_length = cfg.data_config.train_length
        ax_long = subfigs_long[model_idx, 0].add_subplot(1, 1, 1)
        ax_long.set_title(titles[model_idx], loc="left")
        plot_long_term(
            ax_long, data_dict["t"], my_p, my_p_pred, train_length, model_color
        )

        # Plot the spectrum
        ax_spectra = subfigs_long[model_idx, 1].add_subplot(1, 1, 1)
        ax_spectra.set_title(titles[3 + model_idx], loc="left")
        inset_x_lim_psd = [dominant_freq - 0.1, dominant_freq + 0.1]
        inset_y_lim_psd = [0, np.max(psd)]
        plot_psd(
            ax_spectra,
            omega,
            psd,
            psd_pred,
            bg_color,
            model_color,
            inset_x_lim=inset_x_lim_psd,
            inset_y_lim=inset_y_lim_psd,
            harmonic_freq=harmonic_freq,
            dominant_freq=dominant_freq,
        )

    fig_long.savefig("figures/figure_6.png")
    fig_w.savefig("figures/figure_7.png")


if __name__ == "__main__":
    main()
