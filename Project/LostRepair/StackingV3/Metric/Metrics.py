# coding:utf-8

import numpy as np
import pandas as pd


def ar_ks(pred, target):
    # 准备数据
    all_prepare = pd.DataFrame({"pred": pred, "target": target})
    all = pd.DataFrame(
        {
            "total": all_prepare.groupby(["pred"])["target"].count(),
            "good": all_prepare.groupby(["pred"])["target"].count() - all_prepare.groupby(["pred"])["target"].sum(),
            "bad": all_prepare.groupby(["pred"])["target"].sum()
        }
    )
    all.reset_index(drop=False, inplace=True)
    # True 的概率由高到底（分数由低到高）
    all.sort_values(by="pred", ascending=False, inplace=True)
    for i in ["total", "good", "bad"]:
        all[i+"Pcnt"] = all[i] / all[i].sum()
        all[i+"CumRate"] = all[i].cumsum() / all[i].sum()
    all.reset_index(drop=True, inplace=True)

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


