from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator, MultipleLocator

import hcta.postprocessing as post
from hcta.utils import config
from hcta.utils.visualizations2 import plot_2d_v2 as plot_2d
from hcta.utils.visualizations2 import surf_v2 as surf


def grid_plot_colored(
    fig,
    var,
    model_paths,
    model_epochs,
    t_plot,
    inset_x_lim,
    inset_y_lim,
    LIM,
    LIM_ERROR,
    fontsize,
    linewidth,
    bg_colors=None,
):
    plt.rcParams["text.usetex"] = True
    # 1: velocity/pressure in 3d
    # 2: velocity/pressure error in 3d
    # 3: velocity/pressure in 2d
    # 4: velocity/pressure error in 2d
    # width_ratios = [1]
    # width_ratios.extend(len(model_paths) * [4])
    width_ratios = [1] * len(model_paths)
    height_ratios = [1, 1, 0.08, 1, 0.15, 1]
    subfigs = fig.subfigures(
        6,
        len(model_paths),
        width_ratios=width_ratios,
        height_ratios=height_ratios,
    )
    var_str = var.lower()
    # txts = [
    #    f"Prediction \n $\hat{{{var_str}}}'$",
    #    f"Error \n ${var_str}'-\hat{{{var_str}}}'$",
    #    f"Prediction \n $\hat{{{var_str}}}'$",
    #    f"Error \n ${var_str}'-\hat{{{var_str}}}'$",
    # ]
    # for i in range(4):
    #    ax0 = subfigs[i, 0].subplots(1, 1)
    #    ax0.set_axis_off()
    #    ax0.text(
    #        0.5,
    #        0.5,
    #        txts[i],
    #        fontsize=fontsize + 20,
    #        horizontalalignment="center",
    #        verticalalignment="center",
    #        rotation=90,
    #    )

    # define a list of letters (from "a" to "z")
    letters = [chr(letter) for letter in range(ord("a"), ord("z") + 1)]

    # use f-strings to format each letter as "(letter)"
    titles = [f"({letter})" for letter in letters]

    for model_idx, model_path in enumerate(model_paths):
        # load config file
        config_path = model_path / "config.yml"
        cfg = config.load_config(config_path)
        # determine the learned angular frequencies
        _ = post.get_first_layer_weights(cfg, model_epochs[model_idx])

        # predict using the loaded model
        (
            split_data_dict_true,
            split_data_dict,
            idx_train,
            pred_dict,
            _,
        ) = post.make_prediction(cfg, model_epochs[model_idx])

        # get the x-axis of plot
        x_train = split_data_dict["x_train"]
        x_train_true = split_data_dict_true["x_train"]

        # find the time index to plot
        t_train_true = split_data_dict_true["t_train"]

        dt_true = t_train_true[1] - t_train_true[0]
        t_idx_true = int(np.ceil(t_plot / dt_true))

        dt = split_data_dict["t_train"][1] - split_data_dict["t_train"][0]
        t_idx = int(np.ceil(t_plot / dt))

        # unravel train indices and pass those that match with the time index we wish to plot
        # for our data P_train and U_train have the same shape, so it doesn't matter whose shape we use to unravel
        # however if they are given on different grids, then the code should also be modified
        idx_train_unraveled = np.unravel_index(
            idx_train, split_data_dict["P_train"].shape
        )
        idx_train_at_t_idx = idx_train_unraveled[1][idx_train_unraveled[0] == t_idx]

        # set if both, just pressure, just velocity was given
        if cfg.data_config.observe == "both":
            Z_given = split_data_dict[var + "_train"]
        elif cfg.data_config.observe == "p" and var == "P":
            Z_given = split_data_dict[var + "_train"]
        elif cfg.data_config.observe == "u" and var == "U":
            Z_given = split_data_dict[var + "_train"]
        else:
            Z_given = None

        # row 1: plot velocity/pressure
        subfigs[0, model_idx].set_facecolor("white")
        subfigs[0, model_idx].suptitle(titles[4 * model_idx], x=0.05, y=1)
        surf(
            subfigs[0, model_idx],
            num_rows=1,
            num_cols=1,
            plt_idx=1,
            X=x_train_true,
            Y=t_train_true,
            ZZ=pred_dict[var + "_train"],
            xlabel="\n$x$",
            ylabel="\n$t$",
            zlabel=f"$\hat{{{var_str}}}'(x,t)$",
            LIM=LIM,
            bg_color="white",
            fontsize=fontsize,
        )

        # row 2: plot velocity/pressure error
        subfigs[1, model_idx].set_facecolor("white")
        subfigs[1, model_idx].suptitle(titles[4 * model_idx + 1], x=0.05, y=1)
        surf(
            subfigs[1, model_idx],
            num_rows=1,
            num_cols=1,
            plt_idx=1,
            X=x_train_true,
            Y=t_train_true,
            ZZ=split_data_dict_true[var + "_train"] - pred_dict[var + "_train"],
            xlabel="\n$x$",
            ylabel="\n$t$",
            zlabel=f"${var_str}'(x,t)-\hat{{{var_str}}}'(x,t)$",
            LIM=LIM_ERROR,
            bg_color="white",
            fontsize=fontsize,
        )

        # row 3: plot velocity/pressure in 2d
        subfigs[3, model_idx].set_facecolor("white")
        subfigs[3, model_idx].suptitle(titles[4 * model_idx + 2], x=0.05, y=1.15)
        ax, legend_list = plot_2d(
            subfigs[3, model_idx],
            num_rows=1,
            num_cols=1,
            plt_idx=1,
            X=x_train,
            X_true=x_train_true,
            Y_idx=t_idx,
            idx_train=idx_train_at_t_idx,
            Y_idx_true=t_idx_true,
            Z=Z_given,
            Z_true=split_data_dict_true[var + "_train"],
            Z_pred=pred_dict[var + "_train"],
            xlabel="$x$",
            zlabel="",
            title=f"$\hat{{{var_str}}}'(x,t = {t_plot})$",
            axis=0,
            add_inset=var == "U",
            inset_x_lim=inset_x_lim,
            inset_y_lim=inset_y_lim,
            add_legend=True,
            fontsize=fontsize,
            linewidth=linewidth,
        )
        plt.gcf().subplots_adjust(bottom=0.15)
        ax.legend(
            legend_list,
            loc="upper left",
            fontsize=fontsize - 2,
            handletextpad=0.2,
            handlelength=0.8,
        )
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(LinearLocator(numticks=5))
        ax.ticklabel_format(useOffset=False, style="scientific", scilimits=(0, 0))
        ax.set_title(f"$\hat{{{var_str}}}'(x,t = {t_plot})$", fontsize=fontsize, pad=55)

        # row 4: plot velocity/pressure error in 2d
        subfigs[5, model_idx].set_facecolor("white")
        subfigs[5, model_idx].suptitle(titles[4 * model_idx + 3], x=0.05, y=1.15)
        ax = plot_2d(
            subfigs[5, model_idx],
            num_rows=1,
            num_cols=1,
            plt_idx=1,
            X=x_train,
            X_true=x_train_true,
            Y_idx=t_idx,
            idx_train=idx_train_at_t_idx,
            Y_idx_true=t_idx_true,
            Z=None,
            Z_true=None,
            Z_pred=split_data_dict_true[var + "_train"] - pred_dict[var + "_train"],
            xlabel="$x$",
            zlabel="",
            title=f"${var_str}'(x,t = {t_plot})-\hat{{{var_str}}}'(x,t = {t_plot})$",
            axis=0,
            add_legend=False,
            fontsize=fontsize,
            linewidth=linewidth,
        )
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(LinearLocator(numticks=5))
        ax.ticklabel_format(useOffset=False, style="scientific", scilimits=(0, 0))
        ax.set_title(
            f"${var_str}'(x,t = {t_plot})-\hat{{{var_str}}}'(x,t = {t_plot})$",
            fontsize=fontsize,
            pad=55,
        )
    return


def main():
    plot_name = "21"
    if plot_name == "11":
        # Best of FNN-GNN grid plot
        # which models to plot
        model_paths = [
            Path("figure_data/figure_11/model_1"),
            Path("figure_data/figure_11/model_2"),
            Path("figure_data/figure_11/model_3"),
            Path("figure_data/figure_11/model_4"),
        ]
        # which epoch of the model to load the weights from
        model_epochs = [None, None, None, None]
        var = "U"
    elif plot_name == "12":
        # GNN regularization grid plot
        # which models to plot
        model_paths = [
            Path("figure_data/figure_12/model_1"),
            Path("figure_data/figure_12/model_2"),
            Path("figure_data/figure_12/model_3"),
            Path("figure_data/figure_12/model_4"),
        ]
        # which epoch of the model to load the weights from
        model_epochs = [None, None, None, None]
        var = "U"
    elif plot_name == "20":
        # partial sensors
        # which models to plot
        model_paths = [
            Path("figure_data/figure_20_21/model_1"),
            Path("figure_data/figure_20_21/model_2"),
            Path("figure_data/figure_20_21/model_3"),
        ]
        # which epoch of the model to load the weights from
        model_epochs = [None, None, None, None]
        var = "P"
    elif plot_name == "21":
        # partial sensors
        # which models to plot
        model_paths = [
            Path("figure_data/figure_20_21/model_1"),
            Path("figure_data/figure_20_21/model_2"),
            Path("figure_data/figure_20_21/model_3"),
        ]
        # which epoch of the model to load the weights from
        model_epochs = [None, None, None, None]
        var = "U"
    t_plot = 1.5  # time instance to plot in 2d
    inset_x_lim = [0.5, 0.65]
    inset_y_lim = [0.006, 0.012]
    fontsize = 55
    linewidth = 3.5
    LIM = None  # 0.02
    LIM_ERROR = None  # 0.0025
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = fontsize

    fig = plt.figure(figsize=(40, 38))
    grid_plot_colored(
        fig,
        var,
        model_paths,
        model_epochs,
        t_plot=t_plot,
        inset_x_lim=inset_x_lim,
        inset_y_lim=inset_y_lim,
        LIM=LIM,
        LIM_ERROR=LIM_ERROR,
        fontsize=fontsize,
        linewidth=linewidth,
    )

    fig.savefig(f"figures/figure_{plot_name}.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
