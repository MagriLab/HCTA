from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from sklearn.preprocessing import StandardScaler

import hcta.experiment as exp
import hcta.postprocessing as post
from hcta.utils import config
from hcta.utils import preprocessing as pp
from hcta.utils import signals
from hcta.utils.config import DataConfig


def make_prediction(cfg, model_epoch, rijke, input, data_dict):
    my_nn = post.load_network(cfg, epoch=model_epoch, rijke=rijke)

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


def plot_long_term(ax, t, y, y_pred, train_length, xlim, model_color):
    # Plot the long term prediction
    rect = patches.Rectangle([0, -11], width=train_length, height=24, facecolor="red")
    ax.add_patch(rect)
    ax.plot(0, 0, color="grey", linewidth=3.5)
    ax.plot(0, 0, color="darkorange", linewidth=3.5)
    ax.plot(t, y, color="grey", linewidth=1)
    ax.plot(t, y_pred, color="darkorange", linewidth=0.5, linestyle="-")
    ax.set_ylim([-11, 13])

    ax.set_xlabel("$t$")
    ax.set_ylabel("$p'(x,t)$")
    ax.set_xlim(xlim)
    ax.legend(
        ["Train", "True", "Prediction"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        columnspacing=0.65,
        handletextpad=0.25,
        handlelength=1.5,
    )


def plot_short_term(ax, t, y, t_true, y_true, y_pred, model_color, xlim):
    linewidth = 3.5
    ax.scatter(t, y, color="red", s=100, zorder=2, edgecolor="black")
    ax.plot(t_true, y_true, color="silver", linewidth=5 * linewidth, zorder=1)
    ax.plot(
        t_true, y_pred, color=model_color, linewidth=linewidth, linestyle="-", zorder=1
    )
    ax.set_xlim(xlim)
    ax.set_ylabel("$p'(x,t)$")
    ax.grid()
    ax.legend(
        ["Train", "True", "Prediction"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
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
    # Plot the power spectral density
    linewidth = 3.5
    ax.plot(omega, psd, color="silver", linewidth=5 * linewidth)
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
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        columnspacing=0.65,
        handletextpad=0.25,
        handlelength=1.5,
    )
    inset_ax = ax.inset_axes([0.5, 0.45, 0.45, 0.45])
    inset_ax.plot(omega, psd, color="silver", linewidth=5 * linewidth)
    inset_ax.plot(
        omega, psd_pred, color=model_color, linewidth=linewidth, linestyle="-"
    )
    inset_ax.set_yticks([])
    inset_ax.set_ylim(inset_y_lim)
    inset_ax.set_xlim(inset_x_lim)
    inset_ax.set_xticks([dominant_freq])
    inset_ax.grid()
    inset_ax.set_facecolor(bg_color)


def main():
    # load true data
    plot_name = "def"
    if plot_name == "abc":
        beta = 6
        model_paths = [Path("figure_data/figure_8/model_1")]
        titles = ["(a)", "(b)", "(c)"]
        inset_y_lim = [0, 5.0]
    elif plot_name == "def":
        beta = 7
        model_paths = [Path("figure_data/figure_8/model_2")]
        titles = ["(d)", "(e)", "(f)"]
        inset_y_lim = [0, 3.5]
    data_config = DataConfig(
        data_type="rijke",
        data_path=f"data/rijke_kings_beta_{beta}_tau_0_2_long.h5",
        standardise=False,
        dt=0.01,
        noise=None,
        train_length=None,
        val_length=None,
        test_length=None,
        batch_size=None,
    )
    data_dict, rijke = exp.load_data_rijke(data_config)
    plot_t_length = 1000
    dt = data_dict["t"][1] - data_dict["t"][0]
    plot_t_idx = int(np.round(plot_t_length / dt))
    data_dict["t"] = data_dict["t"][:plot_t_idx]
    data_dict["P"] = data_dict["P"][:plot_t_idx, :]
    data_dict["U"] = data_dict["U"][:plot_t_idx, :]

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

    # load network
    model_epochs = [None]
    model_colors = ["black"]
    model_names = ["GalNN"]
    inset_x_lim = [6.4, 6.6]
    harmonic_freq = data_omega_psd
    dominant_freq = data_dominant_psd
    fontsize = 55
    # bg_color = [170 / 256, 190 / 256, 210 / 256]
    # bg_color = adjust_lightness(bg_color, 1.1)
    bg_color = "white"
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = fontsize

    for model_idx, (model_path, model_epoch, model_color) in enumerate(
        zip(model_paths, model_epochs, model_colors)
    ):
        fig_long = plt.figure(constrained_layout=True, figsize=(38, 20))
        subfigs_long = fig_long.subfigures(2, 1)
        subfigs_long_1 = subfigs_long[1].subfigures(1, 2)
        # load config
        config_path = model_path / "config.yml"
        cfg = config.load_config(config_path)
        (
            _,
            split_data_dict,
            idx_train,
            _,
            _,
            _,
        ) = post.load_prediction_data(cfg)
        idx_train_unraveled = np.unravel_index(
            idx_train, split_data_dict["P_train"].shape
        )
        idx_train_at_x_idx = idx_train_unraveled[0][
            idx_train_unraveled[1] == x_plot_idx
        ]

        # load and preprocess
        # predict
        pred_dict = make_prediction(cfg, model_epoch, rijke, my_input, data_dict)

        # extract the predicted pressure time series
        my_p_pred = pred_dict["P"][:, x_plot_idx]

        # cut the predicted pressure time series
        my_p_pred_periodic = my_p_pred[start_pk_idx:end_pk_idx]

        # find psd of the prediction
        _, psd_pred = signals.power_spectral_density(my_p_pred_periodic, dt)

        # Plot the long term prediction
        train_length = cfg.data_config.train_length
        xlim = [train_length - 4, train_length + 20]
        xlim2 = [0, 800]
        ax_short = subfigs_long[0].add_subplot(1, 1, 1)
        ax_short.set_title(titles[0], loc="left")
        t_train = split_data_dict["t_train"][idx_train_at_x_idx]
        t = data_dict["t"]
        my_p_train = split_data_dict["P_train"][idx_train_at_x_idx, x_plot_idx]
        if plot_name == "chaotic":
            t_lyap = 1 / 0.12
            t = t / t_lyap
            t_train = t_train / t_lyap
            train_length = train_length / t_lyap
            xlim = [xlim[i] / t_lyap for i in range(2)]
            xlim2 = [xlim2[i] / t_lyap for i in range(2)]

        plot_short_term(
            ax_short,
            t_train,
            my_p_train,
            t,
            my_p,
            my_p_pred,
            model_color,
            xlim=xlim,
        )
        if plot_name == "chaotic":
            ax_short.set_xlabel("$t \; [LT]$")
        else:
            ax_short.set_xlabel("$t$")
        ax_long = subfigs_long_1[0].add_subplot(1, 1, 1)
        ax_long.set_title(titles[1], loc="left")
        plot_long_term(ax_long, t, my_p, my_p_pred, train_length, xlim2, model_color)
        if plot_name == "chaotic":
            ax_long.set_xlabel("$t \; [LT]$")
        else:
            ax_long.set_xlabel("$t$")

        # Plot the spectrum
        ax_spectra = subfigs_long_1[1].add_subplot(1, 1, 1)
        ax_spectra.set_title(titles[2], loc="left")
        plot_psd(
            ax_spectra,
            omega,
            psd,
            psd_pred,
            bg_color,
            model_color,
            inset_x_lim,
            inset_y_lim,
            harmonic_freq,
            dominant_freq,
        )

    fig_long.savefig(f"figures/figure_8_{plot_name}.png")

    # plt.show()


if __name__ == "__main__":
    main()
