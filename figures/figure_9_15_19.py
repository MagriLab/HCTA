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
    t_plot = 1.5  # time instance to plot in 2d
    inset_x_lim = [0.5, 0.65]
    inset_y_lim = [0.006, 0.012]
    fontsize = 55
    linewidth = 7.0

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = fontsize

    fig = plt.figure(constrained_layout=True, figsize=(38, 26))
    width_ratios = [1, 1]
    height_ratios = [1, 1]
    subfigs = fig.subfigures(
        2, 2, width_ratios=width_ratios, height_ratios=height_ratios, wspace=0
    )

    titles = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    plot_name = "19"
    if plot_name == "9":
        x_plot = 0.25
        model_paths = [
            Path("figure_data/figure_9/model_1"),
            Path("figure_data/figure_9/model_2"),
            Path("figure_data/figure_9/model_3"),
            Path("figure_data/figure_9/model_4"),
        ]
        model_legends = ["P-FNN", "PI-P-FNN", "GalNN", "PI-GalNN"]
        model_epochs = [None, None, None, None]
        model_colors = ["tab:blue", "tab:cyan", "darkorange", "black"]
        model_linestyles = ["-", "-", "--", "-"]
        model_linewidths = [14, 10, 8.5, 3.5]
    elif plot_name == "15":
        # choose model
        x_plot = 2 / 6
        model_paths = [
            Path("figure_data/figure_15/model_1"),
            Path("figure_data/figure_15/model_2"),
        ]
        model_legends = ["PI-P-FNN", "PI-P-GalNN"]
        model_epochs = [None, None]
        model_colors = ["tab:cyan", "black"]
        model_linestyles = ["-", "-"]
        model_linewidths = [10, 3.5]
    elif plot_name == "19":
        x_plot = 0.25
        model_paths = [
            Path("figure_data/figure_19/model_1"),
            Path("figure_data/figure_19/model_2"),
            Path("figure_data/figure_19/model_3"),
        ]
        model_legends = ["a = 1", "a = 20", "a = 10"]
        model_epochs = [None, None, None, None]
        model_colors = ["tab:blue", "darkorange", "black"]
        model_linestyles = ["-", "--", "-"]
        model_linewidths = [14, 11, 3.5]

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

    if not np.isclose(x_train[x_idx], x_train_true[x_idx_true]):
        Z_p_given = None
        Z_u_given = None

    subfigs[0, 0].suptitle(titles[0], x=0.1, y=1.025)
    plot_2d(
        subfigs[0, 0],
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

    subfigs[0, 1].suptitle(titles[2], x=0.1, y=1.025)
    plot_2d(
        subfigs[0, 1],
        num_rows=1,
        num_cols=1,
        plt_idx=1,
        X=x_train,
        X_true=x_train_true,
        Y_idx=t_idx,
        idx_train=idx_train_at_t_idx,
        Y_idx_true=t_idx_true,
        Z=Z_u_given,
        Z_true=split_data_dict_true["U_train"],
        Z_pred=None,
        Z_pred_list=U_stack_pred_list,
        xlabel="$x$",
        zlabel="",
        title=f"$u'(x,t = {t_plot:.2f})$",
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

    subfigs[1, 0].suptitle(titles[1], x=0.1, y=1.025)
    plot_2d(
        subfigs[1, 0],
        num_rows=1,
        num_cols=1,
        plt_idx=1,
        X=t_train,
        X_true=t_stack_true,
        Y_idx=x_idx,
        idx_train=idx_train_at_x_idx,
        Y_idx_true=x_idx_true,
        Z=Z_p_given,
        Z_true=P_stack_true,
        Z_pred=None,
        Z_pred_list=P_stack_pred_list,
        xlabel="$t$",
        zlabel="",
        title=f"$p'(x = {x_plot:.2f},t)$",
        axis=1,
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

    subfigs[1, 1].suptitle(titles[3], x=0.1, y=1.025)
    plot_2d(
        subfigs[1, 1],
        num_rows=1,
        num_cols=1,
        plt_idx=1,
        X=t_train,
        X_true=t_stack_true,
        Y_idx=x_idx,
        idx_train=idx_train_at_x_idx,
        Y_idx_true=x_idx_true,
        Z=Z_u_given,
        Z_true=U_stack_true,
        Z_pred=None,
        Z_pred_list=U_stack_pred_list,
        xlabel="$t$",
        zlabel="",
        title=f"$u'(x = {x_plot:.2f},t)$",
        axis=1,
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

    fig.savefig(f"figures/figure_{plot_name}.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
