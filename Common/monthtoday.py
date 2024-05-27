import pandas as pd
import numpy as np


def month_to_day(factor):
    print("月频转日频")
    ret = pd.read_csv("path to ret", index_col=0)
    ret.index = pd.to_datetime(ret.index)
    day_factor = ret.copy()[ret.index >= factor.index[0]]
    day_factor.loc[:, :] = np.nan
    day_factor.update(factor)
    day_factor = day_factor.ffill()
    day_factor[ret.isnull()] = np.nan
    return day_factor
