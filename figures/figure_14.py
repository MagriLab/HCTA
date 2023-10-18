import argparse

import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np

import hcta.postprocessing as post
from hcta.utils import config
from hcta.utils.visualizations import adjust_lightness


def set_box_colors(bp, cmap):
    colors = cmap(np.linspace(0, 1, len(bp["boxes"])))
    hatch_list = ["xxxx", "////", "\\\\\\", "||", None]
    for (box_idx, box) in enumerate(bp["boxes"]):
        fill_box_color = colors[box_idx]
        box_color = adjust_lightness(fill_box_color, 0.5)
        plt.setp(bp["boxes"][box_idx], color=box_color, facecolor=fill_box_color)
        box.set(hatch=hatch_list[box_idx], linewidth=5.5)

        # changing color and linewidth of
        # whiskers
        for whisker in bp["whiskers"][2 * box_idx : 2 * box_idx + 2]:
            whisker.set(color=box_color, linewidth=5.5, linestyle=":")

        # changing color and linewidth of
        # caps
        for cap in bp["caps"][2 * box_idx : 2 * box_idx + 2]:
            cap.set(color=box_color)

        # changing color and linewidth of
        # medians
        bp["medians"][box_idx].set(color=box_color, linewidth=5.5)

        # changing style of fliers
        bp["fliers"][box_idx].set(marker="D", color=box_color, alpha=0.5)


def main(args):
    # load config file
    rbst_cfg = config.yaml2dict(args.robustness_config_path)
    noise_std_list = rbst_cfg["data_config"]["noise_std"]
    lambda_pi_list = rbst_cfg["train_config"]["lambda_pi"]
    results_mat = np.empty(
        (len(noise_std_list), len(lambda_pi_list), rbst_cfg["N_runs"])
    )
    for (noise_std_idx, noise_std) in enumerate(noise_std_list):
        for (lambda_pi_idx, _) in enumerate(lambda_pi_list):
            for run_idx in range(rbst_cfg["N_runs"]):
                # configure paths in the final loop
                model_path = (
                    f"noise_{noise_std}/"
                    f"gnn"
                    f"_pi_{lambda_pi_idx}"
                    f"_run_{run_idx+1}"
                )
                results_path = args.results_path / model_path
                # predict on the fine grid and plot the prediction
                # load config file
                config_path = results_path / "config.yml"
                cfg = config.load_config(config_path)
                split_data_dict_true, _, _, pred_dict, _ = post.make_prediction(
                    cfg, epoch=None
                )
                # calculate the data-driven residual on the fine grid
                # R_dd_true = post.calculate_datadriven_residual(
                #    split_data_dict_true, pred_dict
                # )
                # err = R_dd_true["train"]

                stack_P_U = np.hstack(
                    [split_data_dict_true["P_train"], split_data_dict_true["U_train"]]
                )
                stack_P_U_pred = np.hstack([pred_dict["P_train"], pred_dict["U_train"]])
                diff = np.linalg.norm(stack_P_U - stack_P_U_pred, "fro")
                denom = np.linalg.norm(stack_P_U)
                err = 100 * diff / denom

                results_mat[noise_std_idx, lambda_pi_idx, run_idx] = err

    X_axis = np.linspace(0, 2.5 * len(noise_std_list), len(noise_std_list))
    box_width = 0.45
    spacing = 0.15
    total_box_length = len(lambda_pi_list) * box_width
    total_spacing = (len(lambda_pi_list) - 1) * spacing
    total_group_length = total_box_length + total_spacing
    group_length_border = total_group_length - box_width
    group_centres = np.linspace(
        -group_length_border / 2, group_length_border / 2, len(lambda_pi_list)
    )

    fontsize = 55
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = fontsize

    fig = plt.figure(constrained_layout=True, figsize=(28, 14))
    cmap = plt.cm.get_cmap("GnBu")
    for (noise_std_idx, noise_std) in enumerate(noise_std_list):
        bp = plt.boxplot(
            results_mat[noise_std_idx, :, :].T,
            positions=X_axis[noise_std_idx] + group_centres,
            widths=box_width,
            patch_artist=True,
        )
        set_box_colors(bp, cmap)
    legend_list = []
    for lambda_pi in lambda_pi_list:
        legend_list.append(f"$\lambda_M = \lambda_E = {lambda_pi}$")
    plt.legend(
        bp["boxes"],
        legend_list,
        loc="upper left",
        # bbox_to_anchor=(0.5, 1.25),
        # ncol=3,
    )
    plt.xticks(X_axis, noise_std_list)
    plt.xlabel("Noise level ($\%$)")
    plt.ylabel("Relative $\ell_2$-Error ($\%$)")
    # plt.yscale("log")

    fig.savefig("figures/figure_14.png", bbox_inches="tight")

if __name__ == "__main__":
    # read arguments from command line
    parser = argparse.ArgumentParser(
        description="Figure robustness with respect to noise level"
    )
    parser.add_argument(
        "--robustness-config-path",
        type=Path,
        default="src/configs/robustness_noise_config.yml",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default="logs_robustness_noise",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
