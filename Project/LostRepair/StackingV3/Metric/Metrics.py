# coding:utf-8

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from itertools import combinations


def ar_ks_kendall_tau(score, label=None, loan=True):
    score_new = pd.Series(score.reshape((-1, ))) if score is not pd.Series else score.reshape((-1, ))
    score_new = pd.qcut(score_new, 100)
    label_new = pd.Series(label.reshape((-1, ))) if label is not pd.Series else label.reshape((-1, ))

    bin_label = pd.DataFrame({"bin": score_new, "label": label_new})
    bin_label_regroup = pd.concat([bin_label.groupby("bin")["label"].count().to_frame("count"),
                                   bin_label.groupby("bin")["label"].sum().to_frame("bad")], axis=1)
    bin_label_regroup = bin_label_regroup.dropna()
    bin_label_regroup = bin_label_regroup.reset_index(drop=False)

    bin_label_regroup["good"] = bin_label_regroup["count"] - bin_label_regroup["bad"]
    bin_label_regroup["bad_rate"] = bin_label_regroup["bad"] / bin_label_regroup["count"]
    bin_label_regroup["bad_cum_rate"] = bin_label_regroup["bad"].cumsum() / bin_label_regroup["bad"].sum()
    bin_label_regroup["good_cum_rate"] = bin_label_regroup["good"].cumsum() / bin_label_regroup["good"].sum()

    # distribution
    if label is not None and loan:
        score_new_for_distribution = pd.Series(score.reshape((-1,))) if score is not pd.Series else score.reshape((-1,))
        score_new_for_distribution = pd.cut(score_new_for_distribution, 50)
        label_new_for_distribution = pd.Series(label.reshape((-1,))) if label is not pd.Series else label.reshape((-1,))
        bin_label_for_distribution = pd.DataFrame(
            {"bin": score_new_for_distribution, "label": label_new_for_distribution})

        bin_label_regroup_for_distribution = pd.concat(
            [bin_label_for_distribution.groupby("bin")["label"].count().to_frame("count"),
             bin_label_for_distribution.groupby("bin")["label"].sum().to_frame("bad")], axis=1)
        bin_label_regroup_for_distribution = bin_label_regroup_for_distribution.dropna()
        bin_label_regroup_for_distribution = bin_label_regroup_for_distribution.reset_index(drop=False)
        # 分数区间
        x_ticks = bin_label_regroup_for_distribution["bin"]
        bin_label_regroup_for_distribution["bin"] = [i for i in range(len(bin_label_regroup_for_distribution["bin"]))]
        bin_label_regroup_for_distribution["bad rate"] = bin_label_regroup_for_distribution["bad"] / bin_label_regroup_for_distribution["count"]
        bin_label_regroup_for_distribution = bin_label_regroup_for_distribution[["bin", "count", "bad rate"]]

        # 双坐标轴 , 使用分数区间作为 x 轴刻度
        left_ax = plt.gca()
        right_ax = left_ax.twinx()
        left_ax.set_xlabel("score asc")
        left_ax.set_title("score distribution")
        left_ax.tick_params(axis="x", rotation=90)
        right_ax.set_ylim(0, bin_label_regroup_for_distribution["bad rate"].max() * 1.2)
        left_ax.bar(bin_label_regroup_for_distribution["bin"], bin_label_regroup_for_distribution["count"], label="count")
        right_ax.plot(bin_label_regroup_for_distribution["bin"], bin_label_regroup_for_distribution["bad rate"], color="y", marker="o", label="line")
        # 双坐标轴设置图例
        bar, bar_label = left_ax.get_legend_handles_labels()
        line, line_label = right_ax.get_legend_handles_labels()
        left_ax.legend(bar+line, bar_label+line_label)
        plt.xticks(bin_label_regroup_for_distribution["bin"], x_ticks)
        plt.show()
    else:
        score_new_for_distribution = pd.Series(score.reshape((-1, ))) if score is not pd.Series else score.reshape((-1, ))
        score_new_for_distribution = pd.cut(score_new_for_distribution, 50)
        label_new_for_distribution = pd.Series(label.reshape((-1, ))) if label is not pd.Series else label.reshape((-1, ))
        bin_label_for_distribution = pd.DataFrame({"bin": score_new_for_distribution, "label": label_new_for_distribution})

        bin_label_regroup_for_distribution = pd.concat([bin_label_for_distribution.groupby("bin")["label"].count().to_frame("count"),
                                                        bin_label_for_distribution.groupby("bin")["label"].sum().to_frame("bad")], axis=1)
        bin_label_regroup_for_distribution = bin_label_regroup_for_distribution.dropna()
        bin_label_regroup_for_distribution = bin_label_regroup_for_distribution.reset_index(drop=False)
        x_ticks = bin_label_regroup_for_distribution["bin"]
        bin_label_regroup_for_distribution["bin"] = [i for i in range(len(bin_label_regroup_for_distribution["bin"]))]
        bin_label_regroup_for_distribution = bin_label_regroup_for_distribution[["bin", "count"]]

        # 使用分数区间作为 x 轴刻度
        ax = plt.gca()
        ax.set_xlabel("score asc")
        ax.set_title("score distribution")
        ax.tick_params(axis="x", rotation=90)
        ax.bar(bin_label_regroup_for_distribution["bin"], bin_label_regroup_for_distribution["count"], label="count")
        plt.xticks(bin_label_regroup_for_distribution["bin"], x_ticks)
        # 设置图例
        ax.legend()
        plt.show()

    # ar
    bad_cum_rate = [0] + list(bin_label_regroup["bad_cum_rate"])
    # (上底 + 下底) * 1 / 2
    area = np.sum([((bad_cum_rate[i] + bad_cum_rate[i+1]) * 0.01) / 2.0 for i in range(len(bad_cum_rate) - 1)]) - 0.5
    area_perfect = (0.99 + 1) * 1 / 2.0 - 0.5
    ar = area / float(area_perfect)

    ar_df = pd.DataFrame({"perfect_bad_cum_rate": [0] + [1] * 100, "bad_cum_rate": bad_cum_rate, "random_bad_cum_rate": list(np.linspace(0, 1, 101))})
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("score asc")
    ax.set_title("AR = " + str(round(ar, 4)))
    ar_df.plot(ax=ax)
    plt.show()

    # ks
    good_cum_rate = [0] + list(bin_label_regroup["good_cum_rate"])
    bad_cum_rate = [0] + list(bin_label_regroup["bad_cum_rate"])
    ks = max([b - g for b, g in zip(bad_cum_rate, good_cum_rate)])

    ks_df = pd.DataFrame({"good_cum_rate": good_cum_rate, "bad_cum_rate": bad_cum_rate})
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("score asc")
    ax.set_title("KS = " + str(round(ks, 4)))
    ks_df.plot(ax=ax)
    plt.show()

    # Kendall's Tau
    bad_rate = bin_label_regroup["bad_rate"]
    concordant_pair = 0
    discordant_pair = 0
    for i, j in list(combinations(bad_rate, 2)):
        if i > j:
            concordant_pair = concordant_pair + 1
        else:
            discordant_pair = discordant_pair + 1
    kendall_tau = (concordant_pair - discordant_pair) / float(len(list(combinations(bad_rate, 2))))

    kendall_tau_series = bad_rate
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("score asc")
    ax.set_ylabel("bad rate")
    ax.set_title("Kendall's Tau = " + str(round(kendall_tau, 4)))
    kendall_tau_series.plot()
    plt.show()

    return ar, ks, kendall_tau


if __name__ == "__main__":
    df = pd.read_csv("C:\\Users\\Dell\\Desktop\\week\\FC\\anti_fraud\\data\\anti_fraud_score_label.csv")
    df = df[["score", "label"]]
    ar_ks_kendall_tau(df["score"], df["label"], loan=False)
