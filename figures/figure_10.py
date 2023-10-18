import pickle
from pathlib import Path

import matplotlib.pyplot as plt


def loss_history_plot(fig, model_paths, colors, legend, ylim, linewidth, titles):
    subfigs = fig.subfigures(1, 3)
    axes = 3 * [None]
    for ax_idx in range(3):
        axes[ax_idx] = subfigs[ax_idx].add_subplot(1, 1, 1)
        subfigs[ax_idx].suptitle(titles[ax_idx], x=0.1, y=1.1)

    for model_idx, model_path in enumerate(model_paths):
        history_path = model_path / "history.pickle"
        # open a file, where you stored the pickled data
        file = open(history_path, "rb")
        history = pickle.load(file)
        axes[0].plot(
            history["train_loss"], color=colors[model_idx], linewidth=linewidth
        )
        axes[1].plot(
            history["val_interp_loss"], color=colors[model_idx], linewidth=linewidth
        )
        axes[2].plot(
            history["val_extrap_loss"], color=colors[model_idx], linewidth=linewidth
        )
        file.close()

    for ax in axes:
        ax.set_yscale("log")
        ax.grid(alpha=0.5)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend(
            legend,
            loc="upper right",
            bbox_to_anchor=(1, 0.95),
            # ncol=3,
            # columnspacing= 0.65,
            handletextpad=0.25,
            handlelength=1.5,
            fontsize=45,
        )
    axes[0].set_ylim(ylim["train"])
    axes[1].set_ylim(ylim["train"])
    axes[2].set_ylim(ylim["val"])
    axes[0].set_title("Training")
    axes[1].set_title("Validation (Interpolation)")
    axes[2].set_title("Validation (Extrapolation)")


def main():
    # which models to plot, respective colors, and legend labels
    model_paths_fnn = [
        Path("figure_data/figure_10/model_1"),
        Path("figure_data/figure_10/model_2"),
        Path("figure_data/figure_10/model_3"),
    ]
    colors_fnn = ["lightseagreen", "tab:blue", "tab:olive"]
    legend_fnn = ["ReLU FNN", "sin FNN", "sin-ReLU FNN"]

    # which models to plot, respective colors, and legend labels
    model_paths_gnn = [
        Path("figure_data/figure_10/model_4"),
        Path("figure_data/figure_10/model_5"),
        Path("figure_data/figure_10/model_6"),
    ]
    colors_gnn = ["gold", "tab:orange", "tab:purple"]
    legend_gnn = ["P-GalNN (i)", "P-GalNN (ii)", "P-GalNN (iii)"]

    fontsize = 55
    linewidth = 5.5
    ylim = {"train": [1e-12, 1e-1], "val": [1e-12, 1e-1]}
    fig = plt.figure(constrained_layout=True, figsize=(38, 22))
    subfigs = fig.subfigures(3, 1, height_ratios=[1, 0.08, 1])
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = fontsize

    titles = ["(a)", "(c)", "(e)", "(b)", "(d)", "(f)"]
    loss_history_plot(
        subfigs[0],
        model_paths_fnn,
        colors_fnn,
        legend_fnn,
        ylim,
        linewidth=linewidth,
        titles=titles[:3],
    )
    loss_history_plot(
        subfigs[2],
        model_paths_gnn,
        colors_gnn,
        legend_gnn,
        ylim,
        linewidth=linewidth,
        titles=titles[3:6],
    )

    fig.savefig(f"figures/figure_10.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
