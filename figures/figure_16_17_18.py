from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import hcta.postprocessing as post
from hcta.utils import config
from hcta.utils.visualizations import plot_2d_v2 as plot_2d


def rel_l2_error(true, pred):
    diff = np.linalg.norm(true - pred, "fro")
    denom = np.linalg.norm(true, "fro")
    err = 100 * diff / denom
    return err


def main():

    inset_x_lim = [0, 0.05]
    # inset_y_lim = [-0.002, 0.0005]
    inset_y_lim = [-0.0005, 0.0001]
    fontsize = 55
    linewidth = 7.0

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = fontsize

    fig = plt.figure(constrained_layout=True, figsize=(38, 18))
    width_ratios = [1, 1]
    height_ratios = [1, 1]
    subfigs = fig.subfigures(
        1, 2, width_ratios=width_ratios, height_ratios=None, wspace=0
    )
    subfigs2 = subfigs[1].subfigures(
        2, 1, width_ratios=None, height_ratios=height_ratios, wspace=0
    )
    titles = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    plot_name = "18"
    if plot_name == "16":
        # choose model
        x_plot = 2 / 6
        model_paths = [
            Path("figure_data/figure_16/model_1"),
            Path("figure_data/figure_16/model_2"),
            Path("figure_data/figure_16/model_3"),
        ]
        model_legends = ["PI-P-FNN", "PI-P-GalNN I.", "PI-P-GalNN N.I."]
        model_epochs = [None, None, None]
        model_colors = ["tab:cyan", "darkorange", "black"]
        model_linestyles = ["-", "--", "-"]
        model_linewidths = [8.5, 10, 3.5]
        t_plot = 1.65  # time instance to plot in 2d
    elif plot_name == "17":
        # choose model
        x_plot = 2 / 6
        model_paths = [
            Path("figure_data/figure_17/model_1"),
            Path("figure_data/figure_17/model_2"),
            Path("figure_data/figure_17/model_3"),
        ]
        model_legends = ["PI-P-FNN", "PI-P-GalNN I.", "PI-P-GalNN N.I."]
        model_epochs = [None, None, None]
        model_colors = ["tab:cyan", "darkorange", "black"]
        model_linestyles = ["-", "--", "-"]
        model_linewidths = [8.5, 10, 3.5]
        t_plot = 1.9  # 1.1 time instance to plot in 2d
    elif plot_name == "18":
        # choose model
        x_plot = 2 / 6
        model_paths = [
            Path("figure_data/figure_18/model_1"),
            Path("figure_data/figure_18/model_2"),
            Path("figure_data/figure_18/model_3"),
        ]
        model_legends = ["PI-P-FNN", "PI-P-GalNN I.", "PI-P-GalNN N.I."]
        model_epochs = [None, None, None]
        model_colors = ["tab:cyan", "darkorange", "black"]
        model_linestyles = ["-", "--", "-"]
        model_linewidths = [8.5, 11, 3.5]
        t_plot = 2.0  # 1.1 time instance to plot in 2d
    P_stack_pred_list = []
    U_stack_pred_list = []
    P_pred_list = []
    U_pred_list = []

    for (model_path, model_epoch) in zip(model_paths, model_epochs):
        # load config file
        config_path = model_path / "config.yml"
        cfg = config.load_config(config_path)
        # predict using the loaded model
        (
            split_data_dict_true,
            split_data_dict,
            idx_train,
            pred_dict,
            _,
        ) = post.make_prediction(cfg, model_epoch)
        R_dd_true = post.calculate_datadriven_residual(split_data_dict_true, pred_dict)
        print(R_dd_true)
        err_p_train = rel_l2_error(
            split_data_dict_true["P_train"], pred_dict["P_train"]
        )
        err_p_val = rel_l2_error(split_data_dict_true["P_val"], pred_dict["P_val"])
        err_u_train = rel_l2_error(
            split_data_dict_true["U_train"], pred_dict["U_train"]
        )
        err_u_val = rel_l2_error(split_data_dict_true["U_val"], pred_dict["U_val"])
        print("Rel l2 error, train pressure: ", err_p_train)
        print("Rel l2 error, val pressure: ", err_p_val)
        print("Rel l2 error, train velocity: ", err_u_train)
        print("Rel l2 error, val velocity: ", err_u_val)
        P_pred_list.append(pred_dict["P_train"])
        P_stack_pred = np.vstack([pred_dict["P_train"], pred_dict["P_val"]])
        P_stack_pred_list.append(P_stack_pred)
        U_pred_list.append(pred_dict["U_train"])
        U_stack_pred = np.vstack([pred_dict["U_train"], pred_dict["U_val"]])
        U_stack_pred_list.append(U_stack_pred)

    P_stack_true = np.vstack(
        [split_data_dict_true["P_train"], split_data_dict_true["P_val"]]
    )
    U_stack_true = np.vstack(
        [split_data_dict_true["U_train"], split_data_dict_true["U_val"]]
    )
    # get the x-axis of spatial plot
    x_train = split_data_dict["x_train"]
    x_train_true = split_data_dict_true["x_train"]

    # get the x-axis of temporal plot
    t_train = split_data_dict["t_train"]
    t_train_true = split_data_dict_true["t_train"]
    t_stack_true = np.hstack([t_train_true, split_data_dict_true["t_val"]])

    # find the time index to plot
    dt_true = t_train_true[1] - t_train_true[0]
    t_idx_true = int(np.ceil(t_plot / dt_true))

    dt = t_train[1] - t_train[0]
    t_idx = int(np.ceil(t_plot / dt))

    if not np.isclose(t_train[t_idx], t_train_true[t_idx_true]):
        raise ValueError("Choose different t to plot.")

    # find the spatial index to plot
    dx_true = x_train_true[1] - x_train_true[0]
    x_idx_true = int(np.round((x_plot - x_train_true[0]) / dx_true))

    dx = x_train[1] - x_train[0]
    x_idx = int(np.round((x_plot - x_train[0]) / dx))

    if not np.isclose(x_train[x_idx], x_train_true[x_idx_true]):
        raise ValueError("Choose different x to plot.")

    # unravel train indices and pass those that match with the time index we wish to plot
    # for our data P_train and U_train have the same shape, so it doesn't matter whose shape we use to unravel
    # however if they are given on different grids, then the code should also be modified
    idx_train_unraveled = np.unravel_index(idx_train, split_data_dict["P_train"].shape)
    idx_train_at_t_idx = idx_train_unraveled[1][idx_train_unraveled[0] == t_idx]
    idx_train_at_x_idx = idx_train_unraveled[0][idx_train_unraveled[1] == x_idx]

    # set if both, just pressure, just velocity was given
    if cfg.data_config.observe == "both":
        Z_p_given = split_data_dict["P_train"]
        Z_u_given = split_data_dict["U_train"]
    elif cfg.data_config.observe == "p":
        Z_p_given = split_data_dict["P_train"]
        Z_u_given = None
    elif cfg.data_config.observe == "u":
        Z_p_given = None
        Z_u_given = split_data_dict["U_train"]

    subfigs[0].suptitle(titles[0], x=0.1, y=1.025)
    ax = plot_2d(
        subfigs[0],
        num_rows=1,
        num_cols=1,
        plt_idx=1,
        X=x_train,
        X_true=x_train_true,
        Y_idx=t_idx,
        idx_train=idx_train_at_t_idx,
        Y_idx_true=t_idx_true,
        Z=Z_p_given,
        Z_true=split_data_dict_true["P_train"],
        Z_pred=None,
        Z_pred_list=P_stack_pred_list,
        xlabel="$x$",
        zlabel="",
        title=f"$p'(x,t = {t_plot:.2f})$",
        axis=0,
        add_inset=False,
        inset_x_lim=inset_x_lim,
        inset_y_lim=inset_y_lim,
        add_legend=True,
        fontsize=fontsize,
        linewidth=linewidth,
        pred_colors=model_colors,
        pred_linestyles=model_linestyles,
        pred_linewidths=model_linewidths,
        pred_legends=model_legends,
    )
    ax.set_title(f"$p'(x,t = {t_plot:.2f})$", fontsize=fontsize, pad=190)
    subfigs2[0].suptitle(titles[1], x=0.1, y=1.025)
    ax = subfigs2[0].add_subplot(1, 1, 1)
    ax.plot(
        t_stack_true,
        P_stack_true[:, 0],
        color="silver",
        linestyle="-",
        linewidth=15,
        zorder=1,
    )
    for (P_pred_, pred_color, pred_linestyle, pred_linewidth) in zip(
        P_stack_pred_list, model_colors, model_linestyles, model_linewidths
    ):
        ax.plot(
            t_stack_true,
            P_pred_[:, 0],
            color=pred_color,
            linestyle=pred_linestyle,
            linewidth=pred_linewidth,
            zorder=1,
        )
    # ax.legend(
    #         ["True","PI-P-FNN","PI-P-GalNN I.","PI-P-GalNN N.I."],
    #         loc="upper center",
    #         bbox_to_anchor=(0.5, 1.3), #1.3
    #         ncol=3, #3
    #         fontsize=fontsize,
    #         columnspacing= 0.45,
    #         handletextpad = 0.2,
    #         handlelength = 1 #0.8
    #     )
    ax.set_xlabel("$t$")
    ax.set_title("$p'(x=0,t)$")
    ax.grid()

    subfigs2[1].suptitle(titles[2], x=0.1, y=1.025)
    ax = subfigs2[1].add_subplot(1, 1, 1)
    ax.plot(
        t_stack_true,
        P_stack_true[:, -1],
        color="silver",
        linestyle="-",
        linewidth=15,
        zorder=1,
    )
    for (P_pred_, pred_color, pred_linestyle, pred_linewidth) in zip(
        P_stack_pred_list, model_colors, model_linestyles, model_linewidths
    ):
        ax.plot(
            t_stack_true,
            P_pred_[:, -1],
            color=pred_color,
            linestyle=pred_linestyle,
            linewidth=pred_linewidth,
            zorder=1,
        )
    # ax.legend(
    #         ["True","PI-P-FNN","PI-P-GalNN I.","PI-P-GalNN N.I."],
    #         loc="upper center",
    #         bbox_to_anchor=(0.5, 1.3), #1.3
    #         ncol=3, #3
    #         fontsize=fontsize,
    #         columnspacing= 0.45,
    #         handletextpad = 0.2,
    #         handlelength = 1 #0.8
    #     )
    ax.set_xlabel("$t$")
    ax.set_title("$p'(x=1,t)$")
    ax.grid()

    fig.savefig(f"figures/figure_{plot_name}.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
