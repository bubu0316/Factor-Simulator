import numpy as np
from tqdm import tqdm


def p_l_s(close, limit, stopping, data, weight):
    """
    此函数用于将权重的Dataframe做涨跌停的处理
    注意：传入的五个Dataframe的日期和股票要对齐
    :param close: close的Dataframe
    :param limit: limit的Dataframe
    :param stopping: stopping的Dataframe
    :param data: data的Dataframe
    :param weight: weight的Dataframe
    :return: 做完涨跌停处理后的weight
    """
    print("涨跌停处理")
    close = close.copy().iloc[1:, ]
    limit = limit.copy().iloc[1:, ]
    stopping = stopping.copy().iloc[1:, ]
    close = close.astype(float)
    limit = limit.astype(float)
    stopping = stopping.astype(float)
    data = data.astype(float)
    data = data.copy().iloc[1:, ]
    limit = close >= limit
    stopping = close <= stopping
    weight = weight.astype(float)
    for i in tqdm(range(weight.shape[0] - 1)):
        flag = (((weight.iloc[i + 1] > (weight.iloc[i] * (1 + data.iloc[i]))) & limit.iloc[i]) |
                ((weight.iloc[i + 1] < (weight.iloc[i] * (1 + data.iloc[i]))) & stopping.iloc[i]))
        weight.iloc[i + 1] = np.where(flag, weight.iloc[i] * (1 + data.iloc[i]), weight.iloc[i + 1])
        if weight.iloc[i + 1][~flag].sum() != 0:
            k = (1 - (weight.iloc[i + 1][flag]).sum()) / weight.iloc[i + 1][~flag].sum()
            weight.iloc[i + 1] = np.where(~flag, weight.iloc[i + 1] * k, weight.iloc[i + 1])
    return weight
