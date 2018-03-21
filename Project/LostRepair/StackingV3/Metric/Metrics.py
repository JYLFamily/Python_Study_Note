# coding:utf-8

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from itertools import combinations


def ar_ks_kendall_tau(score, label):
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
    ax.set_title("AR")
    ar_df.plot(ax=ax)
    plt.show()

    print("AR: " + str(round(ar, 4)))

    # ks
    good_cum_rate = [0] + list(bin_label_regroup["good_cum_rate"])
    bad_cum_rate = [0] + list(bin_label_regroup["bad_cum_rate"])
    ks = max([b - g for b, g in zip(bad_cum_rate, good_cum_rate)])

    ks_df = pd.DataFrame({"good_cum_rate": good_cum_rate, "bad_cum_rate": bad_cum_rate})
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("score asc")
    ax.set_title("KS")
    ks_df.plot(ax=ax)
    plt.show()

    print("KS: " + str(round(ks, 4)))

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
    ax.set_title("Kendall's Tau")
    kendall_tau_series.plot()
    plt.show()

    print("Kendall's Tau: " + str(round(kendall_tau, 4)))

    return ar, ks, kendall_tau


if __name__ == "__main__":
    df = pd.read_csv("C:\\Users\\Dell\\Desktop\\week\\FC\\score_card\\yunyingshang\\before\\data\\oot_loan_ha_proba_final.csv")
    df["proba"] = df["proba"].apply(lambda x: 481.8621881 - 28.85390082 * np.log(x/(1-x)))
    ar_ks_kendall_tau(df["proba"], df["user_label"])