import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import numpy as np
import scipy.stats

matplotlib.use("agg")


WHICH_EXP = {
    "Logitech": ["Exposure 1", "Exposure 10", "Exposure 25"],
    "Axis": ["Exposure 32", "Exposure 100", "Exposure 200"],
}

CAM_CONF = {
    "Logitech": {
        "e": ["Exposure 1", "Exposure 10", "Exposure 25"],
        "s": "ms",
        "f": 30,
        "etus": 100,
        "c": ["#142850", "#00909e", "#a6dcef", "#27496d"],
        "nrows": 720,
    },
    "Axis": {
        "e": ["Exposure 32", "Exposure 100", "Exposure 1000"],
        "s": "us",
        "f": 25,
        "etus": 1,
        "c": ["#2b580c", "#639a67", "#bac964"],
        "nrows": 2160,
    },
}


M_DISPLAY_NAME = {
    "faster_rcnn_inception_v2_coco_2018_01_28": "FRCNN",
    "ssd_inception_v2_coco_2018_01_28": "SSD",
}

COLORS = ["#142850", "#a6dcef", "#2b580c", "#bac964"]


def confidence_interval(data, confidence=0.99):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return h


def freq_to_num(freq):
    """Frequency string to frequency integer Hz."""
    return int(freq[:-2])


def freq_to_dn(freq):
    """Rounds to closest multiple of 25."""
    return str(int(freq_to_num(freq) // 25 * 25)) + "Hz"


def exp_to_num(exp):
    """Extracts integer from exposure string (e.g, "Exposure 32" -> 32)."""
    return exp.split(" ")[-1]


def figure6_no_of_affected_rows_comparison():
    """Generates Figure 6 in the paper (ACSAC)"""

    df_logitech = pd.read_csv(
        "/home/data/results/profiling/Logitech_final.csv", index_col=0
    )
    df_axis = pd.read_csv("/home/data/results/profiling/Axis_final.csv", index_col=0)

    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
    fsize = 40

    for i, (cam_id, sdf) in enumerate(
        zip(["Logitech", "Axis"], [df_logitech, df_axis])
    ):
        for j, exp in enumerate(WHICH_EXP[cam_id]):
            tdf = sdf[sdf["exposure"] == exp]
            exp_num = int(exp.split(" ")[-1])
            exp_us = CAM_CONF[cam_id]["etus"] * exp_num
            N_min, N_max, theoretical_x = [], [], []
            tdf = tdf.sort_values(by=["duty_cycle"])
            dcles = np.sort(tdf.duty_cycle.unique())
            for dc in dcles:
                t_on = 1.0 / CAM_CONF[cam_id]["f"] * dc / 100 * 1000 * 1000  # us
                t_exp = exp_us
                delta_rst = (
                    1000 / (CAM_CONF[cam_id]["f"] * CAM_CONF[cam_id]["nrows"]) * 1000
                )
                nmin = np.ceil(t_exp / delta_rst) + np.ceil(t_on / delta_rst) - 2
                nmax = np.ceil(t_exp / delta_rst) + np.ceil(t_on / delta_rst)
                N_min.append(nmin)
                N_max.append(nmax)
                theoretical_x.append(t_on / 1000)

            theoretical_x = np.array(theoretical_x)
            N_min, N_max = np.array(N_min), np.array(N_max)
            d1, m1, s1 = (
                tdf["duty_cycle"],
                tdf["affected_rows_mean"],
                tdf["affected_rows_std"],
            )

            d1 = 1 / CAM_CONF[cam_id]["f"] * d1.values / 100 * 1000

            title = (
                "t exp = %d ms" % (exp_num) if i == 0 else "t exp = %d us" % (exp_num)
            )
            ax[i, j].set_title(title, fontsize=fsize)
            ax[i, j].fill_between(theoretical_x, N_min, N_max, alpha=0.75, zorder=0)
            ax[i, j].plot(
                theoretical_x,
                (N_min + N_max) / 2,
                linewidth=3,
                zorder=-1,
                alpha=0.75,
                label="Model",
            )
            ax[i, j].errorbar(
                d1,
                m1,
                yerr=s1,
                linewidth=0,
                marker="x",
                markersize=0,
                ecolor="k",
                elinewidth=3,
                capsize=10,
            )
            ax[i, j].scatter(
                d1, m1, linewidth=2, marker="o", s=150, label="Actual", c="tab:red"
            )
            ax[i, j].grid(True)
            if i == 0:
                ax[i, j].set_xlim((0, 17))
                ax[i, j].set_ylim((0, 501))
            if i == 1:
                ax[i, j].set_ylim((0, 701))
                ax[i, j].set_xlim((-0.5, 10.5))
            if j > 0:
                ax[i, j].set_yticklabels([])

    txt0 = ax[0, 0].text(-9, -450, "# Rows Affected", rotation=90, fontsize=fsize)
    _ = ax[0, 0].text(-6, 100, "Logitech", rotation=90, fontsize=fsize)
    _ = ax[1, 0].text(-3, 20, "Axis", rotation=90, fontsize=fsize)
    txt3 = ax[1, 1].text(0.13, -37.5, "t on (ms)", fontsize=fsize)

    plt.subplots_adjust(hspace=0.6, wspace=0.075)
    lgnd = ax[0, 0].legend(ncol=2, bbox_to_anchor=(2.5, 1.85), fontsize=fsize * 0.9)
    matplotlib.rcParams.update({"font.size": fsize})
    fig.savefig(
        "./tmp_figures/figure6_no_of_affected_rows_comparison.pdf",
        bbox_extra_artists=(lgnd, txt3, txt0),
        bbox_inches="tight",
    )


def _get_d(ton, texp, d_trst):
    texp_true, texp_est = np.copy(texp), np.copy(texp)
    X, Y = np.meshgrid(texp_true, texp_est)
    true_N = np.ceil(X / d_trst) + np.ceil(ton / d_trst)
    est_N = np.ceil(Y / d_trst) + np.ceil(ton / d_trst)
    Z = true_N / est_N
    line_x, line_y = [], []
    for n in range(Z.shape[0]):
        line_x.append(X[n, n])
        line_y.append(Y[n, n])
    Zline = np.ones(shape=(len(line_x)))
    return (X, Y, Z), (line_x, line_y, Zline), est_N, true_N


def figure7_incorrect_t_exposure():
    """Generates Figure 7 in the paper (ACSAC)"""
    fig, ax = plt.subplots(ncols=2, subplot_kw={"projection": "3d"})

    ton = 500
    texp = np.arange(100, 2500, 120)
    d_trst = 1 / (CAM_CONF["Logitech"]["nrows"] * CAM_CONF["Logitech"]["f"]) * 1e6
    surface, line, est_N, true_N = _get_d(ton, texp, d_trst)

    min_z = surface[-1].min()
    max_z = surface[-1].max()

    f = surface[-1].flatten()
    print(
        "Logitech, in [0.5,2.0]: {}".format(
            f[(f < 2.0) & (f > 0.5)].shape[0] / f.shape[0] * 100
        )
    )
    print("Logitech, min_z: {}".format(surface[-1].min()))

    max_est_i = np.argmax(surface[-1].flatten())
    max_est = est_N.flatten()[max_est_i]
    max_true = true_N.flatten()[max_est_i]

    xticks = [
        0,
        1000,
        2000,
    ]
    zticks = [0, 2, 4, 6]

    fs1 = 11
    fs2 = 13

    surf = ax[0].plot_surface(*surface, cmap=cm.coolwarm, antialiased=False)
    ax[0].plot3D(*line, "black", linewidth=0.5, zorder=1000)
    ax[0].set_title("Logitech (t_on = %d us)" % ton, fontsize=fs2)
    pmax = (
        surface[0].flatten()[max_est_i],
        surface[1].flatten()[max_est_i],
        surface[2].flatten()[max_est_i],
    )
    ax[0].text(
        *pmax, "N_o=%d, est N_o=%d" % (max_true, max_est), size=fs1, zorder=1, color="k"
    )
    ax[0].scatter3D(*pmax, marker="o", c="black")
    ax[0].set_xticks(xticks)
    ax[0].set_yticks(xticks)
    ax[0].set_xticklabels(xticks, fontsize=fs1)
    ax[0].set_yticklabels(xticks, fontsize=fs1)
    ax[0].set_xlabel("t_exp", fontsize=fs2)
    ax[0].set_ylabel("est t_exp", fontsize=fs2)

    ton = 200
    texp = texp = np.arange(32, 1000, 50)
    d_trst = 1 / (CAM_CONF["Axis"]["nrows"] * CAM_CONF["Axis"]["f"]) * 1e6
    surface, line, est_N, true_N = _get_d(ton, texp, d_trst)
    xticks = [
        0,
        500,
        1000,
    ]

    max_est_i = np.argmax(surface[-1].flatten())
    max_est = est_N.flatten()[max_est_i]
    max_true = true_N.flatten()[max_est_i]

    ax[1].plot3D(*line, "black", linewidth=0.5, zorder=1000)
    ax[1].set_title("Axis (t_on = %d us)" % ton, fontsize=fs2)
    pmax = (
        surface[0].flatten()[max_est_i],
        surface[1].flatten()[max_est_i],
        surface[2].flatten()[max_est_i],
    )
    ax[1].text(
        *pmax, "N_o=%d, est N_o=%d" % (max_true, max_est), size=fs1, zorder=1, color="k"
    )
    ax[1].scatter3D(*pmax, marker="o", c="black")
    ax[1].set_xticks(xticks)
    ax[1].set_yticks(xticks)
    ax[1].set_xticklabels(xticks, fontsize=fs1)
    ax[1].set_yticklabels(xticks, fontsize=fs1)
    ax[1].set_xlabel("t_exp", fontsize=fs2)
    ax[1].set_ylabel("est t_exp", fontsize=fs2)
    ax[1].set_zlim(min_z, max_z)

    min_z = min(surface[-1].min(), min_z)
    max_z = max(surface[-1].max(), max_z)
    ax[0].set_zlim(min_z, max_z)
    ax[0].set_zticks(zticks)
    ax[0].set_zticklabels(
        [""] + list(map(lambda x: "x{:d}".format(x), zticks[1:])), fontsize=fs1
    )

    ax[1].set_zlim(min_z, max_z)
    ax[1].set_zticks(zticks)
    ax[1].set_zticklabels(
        [""] + list(map(lambda x: "x{:d}".format(x), zticks[1:])), fontsize=fs1
    )
    ax[1].set_zlabel(
        "Distortion Size Increase (N_o/est N_o)",
        fontsize=fs2,
        rotation=0,
    )

    f = surface[-1].flatten()
    print(
        "Axis, % under 2.0: {}".format(
            f[(f < 2.0) & (f > 0.5)].shape[0] / f.shape[0] * 100
        )
    )
    print("Axis, min_z: {}".format(surface[-1].min()))

    ax[0].view_init(30, 135)
    ax[1].view_init(30, 135)

    plt.subplots_adjust(left=0.0, right=0.87)

    plt.savefig("./tmp_figures/figure7_incorrect_t_exposure.pdf", bbox_inches="tight")


def load_object_detection_results():
    df_bdd100k = pd.read_csv("/home/data/results/object_detection/bdd100k_final.csv")
    df_virat = pd.read_csv("/home/data/results/object_detection/virat_final.csv")
    df = pd.concat([df_bdd100k, df_virat], axis=0)

    df["total"] = df["hidden"] + df["unaltered"] + df["misplaced"]
    df["perc_hidden"] = df["hidden"] / df["total"] * 100
    df["perc_misplaced"] = df["misplaced"] / df["total"] * 100
    df["perc_appeared"] = df["appeared"] / df["total"] * 100
    return df


def figure9_frequency_comparison():
    df = load_object_detection_results()

    grouped_df = df.groupby(["frequency", "exposure", "model", "video_name"]).agg(
        {"perc_hidden": ["mean", "std"]}
    )

    exposures = ["Exposure 32", "Exposure 75", "Exposure 200"]
    exposures_t = [
        "t_exp=32us",
        "t_exp=75us",
        "t_exp=200us",
    ]
    frequencies = ["25Hz", "259Hz", "511Hz", "750Hz"]
    model_names = [
        "faster_rcnn_inception_v2_coco_2018_01_28",
        "ssd_inception_v2_coco_2018_01_28",
    ]

    nrows = len(model_names)
    ncols = len(exposures)
    hatches = ["/", "\\", "\\\\", "//"]
    fsize = 28
    fig, ax = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 3), sharey=True
    )
    ax[0, 0].text(-2.7, -80, "% Hidden Objects", fontsize=fsize, rotation=90)

    for rowi, model_name in enumerate(model_names):
        for coli, e in enumerate(exposures):

            if rowi == 0:
                ax[rowi, coli].set_title(exposures_t[coli], fontsize=fsize)
            if coli == 0:
                ax[rowi, coli].set_ylabel(M_DISPLAY_NAME[model_name], fontsize=fsize)

            ax[rowi, coli].set_xticks([])
            for i, freq in enumerate([25, 259, 511, 750]):
                mean = grouped_df.loc[(freq, int(exp_to_num(e)), model_name)][
                    "perc_hidden"
                ]["mean"].mean()
                std = grouped_df.loc[(freq, int(exp_to_num(e)), model_name)][
                    "perc_hidden"
                ]["mean"].std()
                ax[rowi, coli].bar(
                    [i],
                    [mean],
                    width=1,
                    yerr=std,
                    label=freq_to_dn(frequencies[i]),
                    hatch=hatches[i],
                    capsize=2,
                    alpha=0.85,
                    color=COLORS[i],
                )
            ax[rowi, coli].set_ylim(0, 100)
            ax[rowi, coli].grid(True, linestyle="--", linewidth=1)
            ax[rowi, coli].set_yticks([0, 50, 100])
            ax[rowi, coli].set_yticklabels(
                map(lambda x: "%d" % x, [0, 50, 100]), fontsize=fsize
            )

    ax[0, 0].legend(
        ncol=len(frequencies) + 1,
        bbox_to_anchor=(2.7, 1.7),
        fontsize=fsize,
        handletextpad=0.3,
        columnspacing=1,
    )
    matplotlib.rcParams.update({"font.size": fsize})
    plt.subplots_adjust(hspace=0.2, wspace=0.1)
    plt.savefig("tmp_figures/figure9_frequency_comparison.pdf", bbox_inches="tight")


def figure10_effect_exposure():
    df = load_object_detection_results()

    grouped_df = df.groupby(["frequency", "exposure", "model", "duty_cycle"]).agg(
        {
            "perc_hidden": ["mean", "std", confidence_interval],
            "perc_appeared": ["mean", "std", confidence_interval],
            "perc_misplaced": ["mean", "std", confidence_interval],
        }
    )

    freq_num = 750.0

    exposures = ["Exposure 32", "Exposure 75", "Exposure 200"]
    exposures_t = list(
        map(
            lambda x: "t_exp=%dus" % (int(x.split(" ")[-1])),
            exposures,
        )
    )
    duty_cycles = np.array([0.1, 1.0, 5, 10, 20, 40])
    xticks = 1 / freq_num * duty_cycles / 100 * 1000 * 1000  # these are us
    tickstp = [0, 4, 5]

    xtickslabels = list(
        map(lambda x: ("%.1f" % x[1]) if x[0] in tickstp else "", enumerate(xticks))
    )
    model_names = [
        "ssd_inception_v2_coco_2018_01_28",
        "faster_rcnn_inception_v2_coco_2018_01_28",
    ]

    nrows = len(model_names)
    ncols = len(exposures)
    fsize = 28

    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 4, nrows * 3),
        sharey=True,
        sharex=True,
    )
    ax[0, 0].set_ylim((0, 100))
    ax[0, 0].set_yticks([0, 25, 50, 75, 100])
    ax[0, 0].set_xticks(xticks)

    _ = ax[0, 0].text(-400, -50, "% of Objects", fontsize=fsize, rotation=90)
    _ = ax[0, 0].text(840, -170, "t_on (us)", fontsize=fsize)

    for rowi, model_name in enumerate(model_names):
        for coli, e in enumerate(exposures):
            if rowi == 0:
                ax[rowi, coli].set_title(exposures_t[coli], fontsize=fsize)
            if coli == 0:
                ax[rowi, coli].set_ylabel(M_DISPLAY_NAME[model_name], fontsize=fsize)

            df_select = grouped_df.loc[(750, int(exp_to_num(e)), model_name)].loc[
                duty_cycles.tolist()
            ]
            m1 = df_select["perc_hidden"]["mean"].values
            std1 = df_select["perc_hidden"]["confidence_interval"].values
            m2 = df_select["perc_misplaced"]["mean"].values
            std2 = df_select["perc_misplaced"]["confidence_interval"].values
            m3 = df_select["perc_appeared"]["mean"].values
            std3 = df_select["perc_appeared"]["confidence_interval"].values

            ax[rowi, coli].plot(
                xticks, m1, marker="x", markersize=10, label="Hidden", c=COLORS[0]
            )
            ax[rowi, coli].plot(
                xticks, m2, marker="o", markersize=10, label="Misplaced", c=COLORS[-1]
            )
            ax[rowi, coli].plot(
                xticks, m3, marker="s", markersize=10, label="Appeared", c=COLORS[2]
            )
            ax[rowi, coli].grid(
                True,
                linestyle="--",
            )
            ax[rowi, coli].fill_between(
                xticks, m1 - std1, m1 + std1, alpha=0.5, color=COLORS[0]
            )
            ax[rowi, coli].fill_between(
                xticks, m2 - std2, m2 + std2, alpha=0.5, color=COLORS[-1]
            )
            ax[rowi, coli].fill_between(
                xticks, m3 - std3, m3 + std3, alpha=0.5, color=COLORS[2]
            )

            ax[rowi, coli].set_yticklabels(
                ["", "25", "50", "75", "100"], fontsize=fsize * 0.9
            )
            ax[rowi, coli].set_xticklabels(xtickslabels, fontsize=fsize * 0.9)

    _ = ax[0, ncols // 2].legend(ncol=4, bbox_to_anchor=(2.25, 1.75), fontsize=fsize)
    matplotlib.rcParams.update({"font.size": fsize})
    plt.savefig("tmp_figures/figure10_effect_exposure.pdf", bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs("./tmp_figures", exist_ok=True)
    figure6_no_of_affected_rows_comparison()
    figure7_incorrect_t_exposure()
    figure9_frequency_comparison()
    figure10_effect_exposure()
