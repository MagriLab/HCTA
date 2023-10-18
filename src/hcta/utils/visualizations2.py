import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator, MultipleLocator


def surf(X, Y, ZZ, ZZ_pred, xlabel, ylabel, zlabel):
    """
    Plot surfaces in 3D given X,Y,Z data

    Args:
    X, Y: X, Y data on 1d grid
    ZZ: Z data on 2d grid
    ZZ_pred: predicted Z data
    """
    [XX, YY] = np.meshgrid(X, Y)  # create 2d grid

    fig = plt.figure(figsize=plt.figaspect(0.3))
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    surf1 = ax1.plot_surface(
        XX,
        YY,
        ZZ,
        rstride=1,
        cstride=1,
        cmap="coolwarm",
        linewidth=0,
        antialiased=False,
    )
    fig.colorbar(surf1, shrink=0.5, aspect=10)
    ax1.set_title(zlabel)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    surf2 = ax2.plot_surface(
        XX,
        YY,
        ZZ_pred,
        rstride=1,
        cstride=1,
        cmap="coolwarm",
        linewidth=0,
        antialiased=False,
    )
    fig.colorbar(surf2, shrink=0.5, aspect=10)
    ax2.set_title(zlabel + " predict")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)

    err = ZZ - ZZ_pred
    ZZ_norm = np.sqrt(np.sum(ZZ**2, axis=1))
    rel_err = 100 * (np.abs(err) / ZZ_norm[:, None])
    # rel_err = 100 * (np.abs(err) / ZZ)

    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    surf3 = ax3.plot_surface(
        XX,
        YY,
        err,
        rstride=1,
        cstride=1,
        cmap="coolwarm",
        linewidth=0,
        antialiased=False,
    )
    fig.colorbar(surf3, shrink=0.5, aspect=10)
    ax3.set_title("Absolute Error")
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel(ylabel)

    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    surf4 = ax4.plot_surface(
        XX,
        YY,
        rel_err,
        rstride=1,
        cstride=1,
        cmap="coolwarm",
        linewidth=0,
        antialiased=False,
    )
    fig.colorbar(surf4, shrink=0.5, aspect=10)
    ax4.set_title("Relative Error %")
    ax4.set_xlabel(xlabel)
    ax4.set_ylabel(ylabel)
    return fig


# Plot functions
def surf_v2(
    fig,
    num_rows,
    num_cols,
    plt_idx,
    X,
    Y,
    ZZ,
    xlabel,
    ylabel,
    zlabel,
    LIM=None,
    bg_color="white",
    fontsize=14,
):
    [XX, YY] = np.meshgrid(X, Y)  # create 2d grid
    ax = fig.add_subplot(
        num_rows, num_cols, plt_idx, projection="3d", facecolor=bg_color, alpha=1
    )
    ax.ticklabel_format(useOffset=False, style="scientific", scilimits=(0, 0))
    ax.dist = 8
    surf1 = ax.plot_surface(
        XX,
        YY,
        ZZ,
        rstride=1,
        cstride=1,
        cmap="coolwarm",
        linewidth=1,
        antialiased=False,
    )
    # ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    cbar = fig.colorbar(surf1, shrink=0.5, aspect=10, pad=0.1)  # pad = 0.2
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.locator_params(nbins=4)
    cbar.outline.set_edgecolor("white")
    cbar.formatter.set_powerlimits((0, 0))

    ax.set_title(zlabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, linespacing=3, fontsize=fontsize)
    ax.set_ylabel(ylabel, linespacing=3, fontsize=fontsize)
    if LIM:
        ax.set_zlim([-LIM, LIM])
        surf1.set_clim([-LIM, LIM])
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    # ax.tick_params(axis="z", labelsize=fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(4))  # 2
    ax.zaxis.set_major_locator(LinearLocator(numticks=5))
    ax.set_zticklabels([])
    return ax


def heatmap(X, Y, ZZ, ZZ_pred, xlabel, ylabel, zlabel):
    """
    Plot heatmap given X,Y,Z data

    Args:
    X, Y: X, Y data on 1d grid
    ZZ: Z data on 2d grid
    ZZ_pred: predicted Z data
    """
    [XX, YY] = np.meshgrid(X, Y)  # create 2d grid

    fig = plt.figure(figsize=plt.figaspect(0.3))
    ax1 = fig.add_subplot(1, 4, 1)
    c1 = ax1.pcolormesh(XX, YY, ZZ, cmap="coolwarm", shading="auto")
    fig.colorbar(c1)
    ax1.set_title(zlabel)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    ax2 = fig.add_subplot(1, 4, 2)
    c2 = ax2.pcolormesh(XX, YY, ZZ_pred, cmap="coolwarm", shading="auto")
    fig.colorbar(c2)
    ax2.set_title(zlabel + " predict")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)

    err = ZZ - ZZ_pred
    ZZ_norm = np.sqrt(np.sum(ZZ**2, axis=1))
    rel_err = 100 * (np.abs(err) / ZZ_norm[:, None])
    # rel_err = 100 * (np.abs(err) / ZZ)

    ax3 = fig.add_subplot(1, 4, 3)
    c3 = ax3.pcolormesh(XX, YY, err, cmap="coolwarm", shading="auto")
    fig.colorbar(c3)
    ax3.set_title("Absolute Error")
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel(ylabel)

    ax4 = fig.add_subplot(1, 4, 4)
    c4 = ax4.pcolormesh(XX, YY, rel_err, cmap="coolwarm", shading="auto")
    fig.colorbar(c4)
    ax4.set_title("Relative Error %")
    ax4.set_xlabel(xlabel)
    ax4.set_ylabel(ylabel)
    return fig


def plot_2d(
    n_plts, X, Y, Y_idx, Z, Z_true, Z_pred, xlabel, zlabel, title, axis, figsize=(24, 6)
):
    fig = plt.figure(figsize=figsize)
    for plt_idx in range(n_plts):
        ax = fig.add_subplot(1, n_plts, plt_idx + 1)
        if axis == 0:
            if Z is not None:
                ax.scatter(X, Z[Y_idx[plt_idx], :])
            if Z_true is not None:
                ax.plot(X, Z_true[Y_idx[plt_idx], :])
            if Z_pred is not None:
                ax.plot(X, Z_pred[Y_idx[plt_idx], :], "--")
        else:
            if Z is not None:
                ax.scatter(X, Z[:, Y_idx[plt_idx]])
            if Z_true is not None:
                ax.plot(X, Z_true[:, Y_idx[plt_idx]])
            if Z_pred is not None:
                ax.plot(X, Z_pred[:, Y_idx[plt_idx]], "--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(zlabel)
        ax.set_title(title.format(Y[Y_idx[plt_idx]]))

        legend_list = []
        if Z is not None:
            legend_list.append("train")
        if Z_true is not None:
            legend_list.append("true")
        if Z_pred is not None:
            legend_list.append("pred")

        ax.legend(legend_list, loc="upper right")
    return fig


def plot_2d_v2(
    fig,
    num_rows,
    num_cols,
    plt_idx,
    X,
    Y_idx,
    X_true,
    Y_idx_true,
    idx_train,
    Z,
    Z_true,
    Z_pred,
    xlabel,
    zlabel,
    title,
    axis,
    Z_pred_list=None,
    add_legend=True,
    add_inset=False,
    inset_x_lim=None,
    inset_y_lim=None,
    fontsize=14,
    linewidth=1,
    pred_colors=None,
    pred_linestyles=None,
    pred_linewidths=None,
    pred_legends=None,
):
    ax = fig.add_subplot(num_rows, num_cols, plt_idx)
    ax.yaxis.offsetText.set_fontsize(fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    base_train_color = "red"
    base_val_color = "blue"
    base_edge_color = "black"
    base_true_color = "lightgrey"
    base_pred_color = "black"
    if axis == 0:
        if Z is not None:
            ax.scatter(
                X[idx_train],
                Z[Y_idx, idx_train],
                color=base_train_color,
                s=80 * linewidth,
                zorder=2,
                marker="o",
            )
            idx_val = [idx for idx in range(len(X)) if idx not in idx_train]
            ax.scatter(
                X[idx_val],
                Z[Y_idx, idx_val],
                color=base_val_color,
                s=80 * linewidth,
                zorder=2,
                marker="o",
                edgecolors=base_edge_color,
            )
        if Z_true is not None:
            ax.plot(
                X_true,
                Z_true[Y_idx_true, :],
                color=base_true_color,
                linewidth=4.5 * linewidth,
                zorder=1,
            )
        if Z_pred is not None:
            ax.plot(
                X_true,
                Z_pred[Y_idx_true, :],
                color=base_pred_color,
                linestyle="-",
                linewidth=linewidth,
                zorder=1,
            )
        if Z_pred_list is not None:
            for (Z_pred_, pred_color, pred_linestyle, pred_linewidth) in zip(
                Z_pred_list, pred_colors, pred_linestyles, pred_linewidths
            ):
                ax.plot(
                    X_true,
                    Z_pred_[Y_idx_true, :],
                    color=pred_color,
                    linestyle=pred_linestyle,
                    linewidth=pred_linewidth,
                    zorder=1,
                )
    else:
        if Z is not None:
            ax.scatter(
                X[idx_train],
                Z[idx_train, Y_idx],
                color=base_train_color,
                s=20 * linewidth,
                zorder=2,
                marker="o",
            )
            idx_val = [idx for idx in range(len(X)) if idx not in idx_train]
            ax.scatter(
                X[idx_val],
                Z[idx_val, Y_idx],
                color=base_val_color,
                s=15 * linewidth,
                zorder=2,
                marker="o",
                edgecolors=base_edge_color,
            )
        if Z_true is not None:
            ax.plot(
                X_true,
                Z_true[:, Y_idx_true],
                color=base_true_color,
                linewidth=3.5 * linewidth,
                zorder=1,
            )
        if Z_pred is not None:
            ax.plot(
                X_true,
                Z_pred[:, Y_idx_true],
                color=base_pred_color,
                linestyle="-",
                linewidth=linewidth,
                zorder=1,
            )
        if Z_pred_list is not None:
            for (Z_pred_, pred_color, pred_linestyle, pred_linewidth) in zip(
                Z_pred_list, pred_colors, pred_linestyles, pred_linewidths
            ):
                ax.plot(
                    X_true,
                    Z_pred_[:, Y_idx_true],
                    color=pred_color,
                    linestyle=pred_linestyle,
                    linewidth=pred_linewidth,
                    zorder=1,
                )
    ax.set_xlabel(xlabel, fontsize=fontsize, linespacing=3)
    ax.set_ylabel(zlabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)  # pad = 55 grid,122 activation
    ax.grid()
    if add_legend is True:
        legend_list = []
        if Z is not None:
            legend_list.append("Train")
            legend_list.append("Val.")
        if Z_true is not None:
            legend_list.append("True")
        if Z_pred is not None:
            legend_list.append("Pred.")
        if Z_pred_list is not None:
            legend_list.extend(pred_legends)

    ax.margins(x=0)

    if add_inset:
        inset_axes1 = ax.inset_axes([0.5, 0.05, 0.45, 0.45])
        if axis == 0:
            if Z is not None:
                inset_axes1.scatter(
                    X[idx_train],
                    Z[Y_idx, idx_train],
                    color=base_train_color,
                    s=80 * linewidth,
                    zorder=2,
                    marker="o",
                )
            if Z_true is not None:
                inset_axes1.plot(
                    X_true,
                    Z_true[Y_idx_true, :],
                    color=base_true_color,
                    linewidth=3.5 * linewidth,
                    zorder=1,
                )
            if Z_pred is not None:
                inset_axes1.plot(
                    X_true,
                    Z_pred[Y_idx_true, :],
                    color=base_pred_color,
                    linestyle="-",
                    linewidth=linewidth,
                    zorder=1,
                )
            if Z_pred_list is not None:
                for (Z_pred_, pred_color, pred_linestyle, pred_linewidth) in zip(
                    Z_pred_list, pred_colors, pred_linestyles, pred_linewidths
                ):
                    inset_axes1.plot(
                        X_true,
                        Z_pred_[Y_idx_true, :],
                        color=pred_color,
                        linestyle=pred_linestyle,
                        linewidth=pred_linewidth,
                        zorder=1,
                    )
        else:
            if Z is not None:
                inset_axes1.scatter(
                    X[idx_train],
                    Z[idx_train, Y_idx],
                    color=base_train_color,
                    s=80 * linewidth,
                    zorder=1,
                    marker="o",
                )
            if Z_true is not None:
                inset_axes1.plot(
                    X_true,
                    Z_true[:, Y_idx_true],
                    color=base_true_color,
                    linewidth=3.5 * linewidth,
                    zorder=2,
                )
            if Z_pred is not None:
                inset_axes1.plot(
                    X_true,
                    Z_pred[:, Y_idx_true],
                    color=base_pred_color,
                    linestyle="-",
                    linewidth=linewidth,
                    zorder=2,
                )
            if Z_pred_list is not None:
                for (Z_pred_, pred_color, pred_linestyle, pred_linewidth) in zip(
                    Z_pred_list, pred_colors, pred_linestyles, pred_linewidths
                ):
                    inset_axes1.plot(
                        X_true,
                        Z_pred_[:, Y_idx_true],
                        color=pred_color,
                        linestyle=pred_linestyle,
                        linewidth=pred_linewidth,
                        zorder=2,
                    )
        inset_axes1.set_xticks([])
        inset_axes1.set_yticks([])
        inset_axes1.set_xlim(inset_x_lim)
        inset_axes1.set_ylim(inset_y_lim)
    if add_legend:
        return ax, legend_list
    else:
        return ax


def loss_history(hist):
    plt.plot(hist["train_loss"])
    plt.plot(hist["val_loss"])
    plt.xlabel("Epochs")
    plt.legend(["Train", "Val"])
    plt.title("Loss history")


def adjust_lightness(color, amount=0.5):
    """Adjust the lightness of the given color"""
    # taken from https://stackoverflow.com/a/49601444
    import colorsys

    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
