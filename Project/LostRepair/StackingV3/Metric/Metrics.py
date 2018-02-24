# coding:utf-8

import numpy as np
import pandas as pd


def ar_ks(pred, target):
    # 准备数据
    all_prepare = pd.DataFrame({"pred": pred, "target": target})
    all = pd.DataFrame(
        {
            "total": all_prepare.groupby(["pred"])["target"].count(),
            "bad": all_prepare.groupby(["pred"])["target"].sum()
        }
    )
    all["good"] = all["total"] - all["bad"]
    all.reset_index(drop=False, inplace=True)
    # True 的概率由高到底
    all.sort_values(by="pred", ascending=False, inplace=True)
    all["totalPcnt"] = all["total"] / all["total"].sum()
    all["badCumRate"] = all["bad"].cumsum() / all["bad"].sum()
    all["goodCumRate"] = all["good"].cumsum() / all["good"].sum()
    all = all[["pred", "total", "bad", "good", "totalPcnt", "badCumRate", "goodCumRate"]]
    all.index = range(len(all))

    # 计算 ar
    ar_list = [1/2 * all.loc[0, "totalPcnt"] * all.loc[0, "badCumRate"]]
    for j in range(1, len(all)):
        ar = 1/2 * all.loc[j, "totalPcnt"] * np.sum(all.loc[j-1:j, "badCumRate"])
        ar_list.append(ar)
    ar = (np.sum(ar_list) - 1/2) / (1/2 * (1 - all["bad"].sum() / all["total"].sum()))
    ar = round(ar, 4)

    # 计算 ks
    all["badCumRate-goodCumRate"] = all["badCumRate"] - all["goodCumRate"]
    ks = round(np.max(all["badCumRate-goodCumRate"]), 4)

    return ar, ks


