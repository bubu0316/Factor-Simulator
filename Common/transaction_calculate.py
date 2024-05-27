def cal_transaction(weight, ret, rate=0.0005):
    weight_t = (ret.shift(1).iloc[1:, ] + 1).mul(weight.shift(1).iloc[1:, ])
    weight_t_1 = weight.iloc[1:, ]
    turnover = weight_t_1.sub(weight_t).abs().sum(axis=1)
    transaction = turnover * rate
    return transaction
