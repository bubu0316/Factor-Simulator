import pandas as pd
import monthtoday


def cal_factor(data, path, date=None, month_to_day_flag=False, amplify=True):
    factor_data = pd.read_csv(path)
    factor_data = factor_data.set_index("date")
    factor_data.index = pd.to_datetime(factor_data.index)
    if date is not None:
        factor_data = factor_data.loc[date:]
    factor_data = factor_data.loc[list(set(data.index).intersection(set(factor_data.index))), list(
        set(data.columns).intersection(set(factor_data.columns)))]
    factor_data = factor_data.sort_index(ascending=True)
    factor_data = factor_data.sort_index(axis=1, ascending=True)
    factor_data = factor_data.loc[(factor_data.fillna(0) != 0).any(axis=1)]
    if amplify:
        factor_data = 100000 * factor_data

    if month_to_day_flag:
        factor_data = monthtoday.month_to_day(factor_data)

    new_data = data.loc[factor_data.index, factor_data.columns].fillna(0)
    return factor_data, new_data
