import argparse
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np

import hcta.postprocessing as post
from hcta.utils import config
from hcta.utils.visualizations import adjust_lightness


def main(args):
    # load config file
    rbst_cfg = config.yaml2dict(args.robustness_config_path)
    N_sensors_list = rbst_cfg["data_config"]["N_sensors"]
    N_g_list = rbst_cfg["model_config"]["N_g"]
    lambda_pi_list = rbst_cfg["train_config"]["lambda_pi"]
    results_mat = np.empty((len(N_sensors_list), len(N_g_list), len(lambda_pi_list)))
    for (N_sensors_idx, N_sensors) in enumerate(N_sensors_list):
        for (N_g_idx, N_g) in enumerate(N_g_list):
            for (lambda_pi_idx, _) in enumerate(lambda_pi_list):
                # configure paths in the final loop
                model_path = (
                    f"sensor_{N_sensors}/"
                    f"gnn"
                    f"_N_g_{N_g}"
                    f"_pi_{lambda_pi_idx}"
                    f"_run_1"
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
                stack_P_U = np.hstack(
                    [split_data_dict_true["P_train"], split_data_dict_true["U_train"]]
                )
                stack_P_U_pred = np.hstack([pred_dict["P_train"], pred_dict["U_train"]])
                diff = np.linalg.norm(stack_P_U - stack_P_U_pred, "fro")
                denom = np.linalg.norm(stack_P_U)
                err = 100 * diff / denom
                results_mat[N_sensors_idx, N_g_idx, lambda_pi_idx] = err
    X_axis = np.arange(len(N_sensors_list))

    # set the bar widths and the spacing between subgroups in a group
    bar_width = 0.1
    spacing = 0.05
    # subgroups are different pi for each N_g
    # groups are different N_g for each sensor

    # find the subgroup centres
    total_subgroup_length = len(lambda_pi_list) * bar_width
    subgroup_length_border = total_subgroup_length - bar_width
    subgroup_centres = np.linspace(
        -subgroup_length_border / 2, subgroup_length_border / 2, len(lambda_pi_list)
    )

    # find the group centres
    total_bar_length = total_subgroup_length * len(N_g_list)
    total_spacing = (len(N_g_list) - 1) * spacing
    total_group_length = total_bar_length + total_spacing
    group_length_border = total_group_length - total_subgroup_length
    group_centres = np.linspace(
        -group_length_border / 2, group_length_border / 2, len(N_g_list)
    )

    fontsize = 55
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = fontsize
    cmap = plt.cm.get_cmap("rainbow")
    N_g_colors = cmap(np.linspace(0, 1, len(N_g_list)))
    fig = plt.figure(constrained_layout=True, figsize=(28, 14))
    hatch_list = [None, "/", "//", "///"]
    for (N_g_idx, N_g) in enumerate(N_g_list):
        for (lambda_pi_idx, lambda_pi) in enumerate(lambda_pi_list):
            plot_label = f"$N_g = {N_g}$, $\lambda_M = \lambda_E = {lambda_pi}$"
            plot_color = adjust_lightness(
                N_g_colors[N_g_idx], 0.75 + lambda_pi_idx * 0.5
            )
            plt.bar(
                X_axis + group_centres[N_g_idx] + subgroup_centres[lambda_pi_idx],
                results_mat[:, N_g_idx, lambda_pi_idx],
                width=bar_width,
                label=plot_label,
                color=plot_color,
                edgecolor="black",
                hatch=hatch_list[lambda_pi_idx],
            )
    plt.xticks(X_axis, N_sensors_list)
    plt.xlabel("Number of sensors")
    plt.ylabel("Relative $\ell_2$-Error ($\%$)")
    plt.yscale("log")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=len(N_g_list),
        columnspacing=0.45,
        fontsize=45,
    )
    plt.grid()
    fig.savefig("figures/figure_13.png")

if __name__ == "__main__":
    # read arguments from command line
    parser = argparse.ArgumentParser(
        description="Figure robustness with respect to number of sensors"
    )
    parser.add_argument(
        "--robustness-config-path",
        type=Path,
        default="src/configs/robustness_sensor_config.yml",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default="logs_robustness_sensor",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
