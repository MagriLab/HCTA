from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

import hcta.postprocessing as post
from hcta.utils import config
from hcta.utils.visualizations2 import plot_2d_v2 as plot_2d
from hcta.utils.visualizations2 import surf_v2 as surf


def get_activations():
    x = np.arange(-np.pi, np.pi, 0.01)
    y_relu = np.maximum(np.zeros_like(x), x)
    y_tanh = np.tanh(x)
    y_sin = np.sin(x)
    y = [y_relu, y_tanh, y_sin]
    return x, y


def grid_plot(
    fig,
    var,
    model_paths,
    model_epochs,
    x_plot,
    LIM,
    fontsize,
    linewidth,
    bg_colors,
):
    plt.rcParams["text.usetex"] = True
    width_ratios = [1] * len(model_paths)
    height_ratios = [1, 0.2, 0.8]
    subfigs = fig.subfigures(
        3,
        len(model_paths),
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=0.01,
    )
    var_str = var.lower()
    var_title = f"$\hat{{{var_str}}}'$"

    titles = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    for idx, model_path in enumerate(model_paths):
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
        ) = post.make_prediction(cfg, model_epochs[idx])
        Z_stack_true = np.vstack(
            [split_data_dict_true[var + "_train"], split_data_dict_true[var + "_val"]]
        )
        Z_stack_pred = np.vstack([pred_dict[var + "_train"], pred_dict[var + "_val"]])

        # get the x-axis of spatial plot
        x_train = split_data_dict["x_train"]
        x_train_true = split_data_dict_true["x_train"]

        # get the x-axis of temporal plot
        t_train = split_data_dict["t_train"]
        t_train_true = split_data_dict_true["t_train"]
        t_stack_true = np.hstack([t_train_true, split_data_dict_true["t_val"]])

        # find the spatial index to plot
        dx_true = x_train_true[1] - x_train_true[0]
        x_idx_true = int(np.round((x_plot - x_train_true[0]) / dx_true))

        dx = x_train[1] - x_train[0]
        x_idx = int(np.round((x_plot - x_train[0]) / dx))

        if np.isclose(x_train[x_idx], x_train_true[x_idx_true]):
            Z = split_data_dict[var + "_train"]
        else:
            Z = None

        # unravel train indices and pass those that match with the time index we wish to plot
        # for our data P_train and U_train have the same shape, so it doesn't matter whose shape we use to unravel
        # however if they are given on different grids, then the code should also be modified
        idx_train_unraveled = np.unravel_index(
            idx_train, split_data_dict["P_train"].shape
        )
        idx_train_at_x_idx = idx_train_unraveled[0][idx_train_unraveled[1] == x_idx]

        # row 1: plot velocity/pressure
        subfigs[0, idx].set_facecolor(bg_colors[0][idx])
        subfigs[0, idx].suptitle(titles[2 * idx], x=0.1, y=0.9)
        ax_surf = surf(
            subfigs[0, idx],
            num_rows=1,
            num_cols=1,
            plt_idx=1,
            X=x_train_true,
            Y=t_stack_true[0:-1:2],
            ZZ=Z_stack_pred[0:-1:2, :],
            xlabel="\n$x$",
            ylabel="\n$t$",
            zlabel=var_title + f"$(x,t)$",
            LIM=LIM,
            bg_color=bg_colors[0][idx],
            fontsize=fontsize,
        )
        # ax_surf.view_init(30, 30)
        # ax_surf.set_title(titles[idx], loc="center", fontsize=28)
        subfigs[2, idx].set_facecolor(bg_colors[1][idx])
        subfigs[2, idx].suptitle(titles[2 * idx + 1], x=0.1, y=1.22)
        ax, legend_list = plot_2d(
            subfigs[2, idx],
            num_rows=1,
            num_cols=1,
            plt_idx=1,
            X=t_train,
            X_true=t_stack_true,
            Y_idx=x_idx,
            idx_train=idx_train_at_x_idx,
            Y_idx_true=x_idx_true,
            Z=Z,
            Z_true=Z_stack_true,
            Z_pred=Z_stack_pred,
            Z_pred_list=None,
            xlabel="$t$\n",
            zlabel="",
            title=var_title + f"$(x = {x_plot:.2f},t)$",
            axis=1,
            add_inset=False,
            inset_x_lim=None,
            inset_y_lim=None,
            add_legend=True,
            fontsize=fontsize,
            linewidth=linewidth,
        )
        ax.legend(
            legend_list,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.35),
            ncol=3,
            fontsize=fontsize,
            columnspacing=0.45,
            handletextpad=0.2,
            handlelength=0.8,
        )
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.yaxis.set_major_locator(MultipleLocator(4))
        ax.set_title(var_title + f"$(x = {x_plot:.2f},t)$", fontsize=fontsize, pad=122)


def activation_plot(fig, bg_colors):
    subfigs = fig.subfigures(1, 3)
    z, phi_z_list = get_activations()
    # titles = ["(a)", "(b)", "(c)"]
    titles = ["ReLU", "tanh", "sine"]
    for (phi_z_idx, phi_z) in enumerate(phi_z_list):
        subfigs[phi_z_idx].set_facecolor(bg_colors[phi_z_idx])
        subfigs[phi_z_idx].add_subplot(1, 1, 1)
        plt.plot(z, phi_z, color="black")
        plt.grid()
        plt.title(titles[phi_z_idx], loc="center", fontsize=28)
        plt.xlabel("$z$")
        plt.ylabel("$\phi(z)$")
        plt.xlim([z[0], z[-1]])


def main():
    x_plot = 0.25  # location to plot in 2d
    fontsize = 55
    linewidth = 3.5
    LIM = None
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = fontsize
    var = "P"
    model_paths = [
        Path("figure_data/figure_5/model_1"),
        Path("figure_data/figure_5/model_2"),
        Path("figure_data/figure_5/model_3"),
        Path("figure_data/figure_5/model_4"),
    ]
    model_epochs = [None, None, None, None]
    bg_colors = [
        [
            "white",
            "white",
            "white",
            "white",
        ],
        [
            "white",
            "white",
            "white",
            "white",
        ],
    ]
    fig1 = plt.figure(figsize=(40, 18))
    grid_plot(
        fig1,
        var,
        model_paths,
        model_epochs,
        x_plot,
        LIM,
        fontsize,
        linewidth,
        bg_colors,
    )
    fig1.savefig(f"figures/figure_5.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
