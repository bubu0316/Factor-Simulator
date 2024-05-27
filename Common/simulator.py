# 导包
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm


# 定义增加微小差异的函数
def add_epsilon_to_duplicates(row, epsilon):
    mask = row.duplicated(keep='first')
    duplicates = row.loc[mask]
    row.loc[mask] = duplicates - np.arange(1, mask.sum() + 1) * epsilon
    return row


# 定义计算统计量的函数
def cal_statis(sr, turnover):
    # 按年份统计
    year = sr.resample('YE').mean().index.year
    statis = pd.DataFrame(np.zeros(shape=[year.size, 5]), index=year,
                          columns=['年化收益率', '年化波动率', '夏普比率', '最大回撤率', '换手率'])
    statis.index = statis.index.set_names(['year'])
    statis['年化收益率'] = list(sr.resample('YE').mean() * 245)
    statis['年化波动率'] = list(sr.resample('YE').std() * (245 ** 0.5))
    statis['夏普比率'] = statis['年化收益率'] / statis['年化波动率']
    draw_down = pd.concat([(1 + sr).cumprod(), (1 + sr).cumprod().expanding().max()], axis=1)
    draw_down.columns = ["current", "max"]
    draw_down['rate'] = (draw_down["max"] - draw_down["current"]) / draw_down["max"]
    statis['最大回撤率'] = list(draw_down['rate'].resample('YE').max())
    statis['换手率'] = list(turnover.resample('YE').mean() * 245)

    total = [sr.mean() * 245, sr.std() * (245 ** 0.5), (sr.mean() * 245) / (sr.std() * (245 ** 0.5)),
             draw_down['rate'].max(), turnover.mean() * 245]
    statis.loc['total'] = total
    return statis


# 构建模拟器类
class Simulator:
    # 原始数据-日收益率
    data = None
    # 处理后的数据
    new_data = None
    # 行业数据
    industry = None
    # 市值数据
    mv = None
    # 因子数据
    factor = None
    # long
    long = None
    # short
    short = None
    # difference : long - short
    difference = None
    # 日换手率
    turnover = None
    # close数据
    close = None
    # 涨停数据
    limit = None
    # 跌停数据
    stopping = None
    # 分组权重
    group_weight = []
    # 分组数据
    group_data = []
    # 时间序列表
    time_series = None
    # delay
    day = 1
    # top换手率
    top_turnover = None
    # long换手率
    long_turnover = None
    # 交易模式
    mode = None

    # 读取原始数据-日收益率
    def read_data(self, mode, day):
        print("读取日收益率")
        self.day = day
        self.mode = mode
        if mode == "close":
            self.data = pd.read_csv("path to ret", index_col=0)
            self.data.index = pd.to_datetime(self.data.index)
        if mode == "open":
            adj_factor = pd.read_csv("path to adj_factor", index_col=0)
            adj_factor.index = pd.to_datetime(adj_factor.index)
            open = pd.read_csv("path to open", index_col=0)
            open.index = pd.to_datetime(open.index)
            adj_open = adj_factor * open
            ret = adj_open.shift(-1).iloc[:-1, :].sub(adj_open.iloc[:-1, :]).div(adj_open.iloc[:-1, :])
            self.data = ret

    # 计算因子及处理原始数据（以每五日的滚动平均收益率为例）
    def calculate_factor(self, func, path, date=None, month_to_day_flag=False, amplify=True):
        print("读取因子并处理原始数据")
        (self.factor, self.new_data) = func(self.data, path, date, month_to_day_flag, amplify)

    # 读取行业数据
    def read_industry(self, path):
        print("读取行业数据")
        self.industry = pd.read_csv(path, index_col=0, low_memory=False)
        self.industry.index = pd.to_datetime(self.industry.index)
        self.industry.index.name = "date"
        # 标准化industry
        self.industry = self.industry.loc[list(set(self.factor.index).intersection(set(self.industry.index))), list(
            set(self.factor.columns).intersection(set(self.industry.columns)))]
        self.industry = self.industry.sort_index(ascending=True)
        self.industry = self.industry.sort_index(axis=1, ascending=True)

    # 对因子数据做数据清洗
    def factor_clean(self):
        print("数据清洗--缺失值处理")
        # 缺失值处理
        # 判断是否在数据时间的初期出现空值，将这些空值替换为特殊字符"?"
        mask = self.factor.copy()
        mask = mask.where(mask.notnull().cumsum() != 0, "?")
        mask.index = self.factor.index
        mask.columns = self.factor.columns
        # 如果一个空值之前的连续长时间（20天）都为空值，则将该空值替换为特殊字符"?"
        mask = mask.where((mask.isnull().rolling(20).sum() != 20) | (mask.notnull()), "?")
        mask.index = self.factor.index
        mask.columns = self.factor.columns
        # 现在真空值为nan，假空值为"?"
        # 计算缺失率
        missing_rate = mask.isnull().mean()
        # 剔除缺失值大于20%的股票
        mask = mask.loc[mask.index, missing_rate[missing_rate <= 0.2].index]
        # 缺失率小于20%的股票，用当天同行业的因子数据的中位数代替
        # 计算各行业的因子值的中位数
        # 将行列索引一致化
        industry = self.industry.copy()
        stock = list(set(mask.columns).intersection(set(industry.columns)))
        date = list(set(mask.index).intersection(set(industry.index)))
        mask = mask.loc[date, stock].sort_index(axis=1)
        industry = industry.loc[date, stock].sort_index(axis=1)
        if np.any(pd.isnull(mask)):
            # 合并数据
            industry = industry.stack()
            mask_nan = pd.to_numeric(mask.stack(), errors="coerce")
            ind_factor = pd.concat([industry, mask_nan], axis=1)
            ind_factor.columns = ["industry", "factor"]
            # 统计每天行业的因子值的中位数
            ind_median = ind_factor.groupby(["date", "industry"])["factor"].median()
            # 统计缺失值的行列索引
            miss = mask.isnull().stack()[mask.isnull().stack()]
            # # 填充缺失值
            nan_index = list(set(miss.index).intersection(set(industry.index)))
            miss = industry.loc[nan_index]
            for i in miss.index:
                mask.loc[i[0], i[1]] = ind_median[(i[0], miss[i])]
        # 将"?"转化为nan
        pd.set_option('future.no_silent_downcasting', True)
        mask = mask.replace(to_replace="?", value=np.nan).infer_objects(copy=False)
        mask = mask.reindex(index=self.factor.index, columns=self.factor.columns)
        self.factor = self.factor.fillna(mask)
        print("数据清洗--异常值处理")
        # 异常值处理
        outlier_process = self.factor.copy()
        # 计算因子值的中位数
        outlier_process["med_factor"] = outlier_process.median(axis=1)
        # 计算绝对中位值
        outlier_process["mad"] = outlier_process.sub(outlier_process["med_factor"], axis=0).abs().median(axis=1)
        # 定义异常值范围
        outlier_process["upper_limit"] = outlier_process["med_factor"] + 3 * 1.4826 * outlier_process["mad"]
        outlier_process["lower_limit"] = outlier_process["med_factor"] - 3 * 1.4826 * outlier_process["mad"]
        # 替换异常值
        upper_limit = self.factor.copy()
        upper_limit = pd.concat([outlier_process["upper_limit"] for _ in upper_limit.columns], axis=1)
        upper_limit.columns = self.factor.columns
        lower_limit = self.factor.copy()
        lower_limit = pd.concat([outlier_process["lower_limit"] for _ in lower_limit.columns], axis=1)
        lower_limit.columns = self.factor.columns
        new_factor = pd.DataFrame(np.where(self.factor.subtract(upper_limit) > 0, upper_limit, self.factor))
        new_factor.index = self.factor.index
        new_factor.columns = self.factor.columns
        new_factor = pd.DataFrame(np.where(new_factor.subtract(lower_limit) < 0, lower_limit, new_factor))
        new_factor.index = self.factor.index
        new_factor.columns = self.factor.columns
        self.factor = new_factor.copy()

    # 读取市值数据
    def read_mv(self, path):
        print("读取市值数据")
        self.mv = pd.read_csv(path, index_col=0, low_memory=False)
        self.mv.index = pd.to_datetime(self.mv.index)
        self.mv.index.name = "date"
        # 标准化mv
        self.mv = self.mv.loc[list(set(self.factor.index).intersection(set(self.mv.index))), list(
            set(self.factor.columns).intersection(set(self.mv.columns)))]
        self.mv = self.mv.sort_index(ascending=True)
        self.mv = self.mv.sort_index(axis=1, ascending=True)

    # 因子中性化
    def factor_neutral(self, mv_=True, ind_=True):
        print("因子中性化")
        mv = self.mv.copy()
        industry = self.industry.copy()
        factor = self.factor.copy()
        # 市值标准化
        mv = np.log10(mv)
        date = industry.index
        industry = industry.stack(dropna=False)
        ind = pd.get_dummies(industry).replace({True: 1, False: 0})
        factor = factor.stack(dropna=False)
        if mv_ & ind_:
            x = pd.concat([mv.stack(dropna=False), ind], axis=1).rename(columns={0: 'mv'})
            notnull_list = list(set(x[x.notnull().all(axis=1)].index).intersection(set(factor[factor.notnull()].index)))
            x = x.loc[notnull_list, :].sort_index(ascending=True)
        else:
            if mv_:
                x = mv.stack(dropna=False)
                notnull_list = list(set(x[x.notnull()].index).intersection(set(factor[factor.notnull()].index)))
                x = x.loc[notnull_list].sort_index(ascending=True)
            elif ind_:
                x = ind
                notnull_list = list(
                    set(x[x.notnull().all(axis=1)].index).intersection(set(factor[factor.notnull()].index)))
                x = x.loc[notnull_list, :].sort_index(ascending=True)
            else:
                return None

        factor = factor.loc[notnull_list].sort_index(ascending=True)
        o = pd.concat([factor, x], axis=1).astype(float).dropna()

        o['resid'] = np.nan
        for i in tqdm(range(len(date))):
            y = o.loc[date[i]].iloc[:, 0]
            ols_model = sm.OLS(np.asarray(y), sm.add_constant(np.asarray(o.loc[date[i]].iloc[:, 1:-1])))
            ols_results = ols_model.fit().resid
            o.loc[date[i], 'resid'] = ols_results

        result = (o.copy().iloc[:, -1].unstack().dropna(how='all').dropna(axis=1, how='all')
                  .reindex(columns=mv.columns).sort_index(ascending=True).sort_index(axis=1, ascending=True))
        self.factor = result

    # 计算long,short及其差值
    def calculate_difference(self, func, progress_limit=False, rate=0.0005, rolling_day=1):
        print("计算long,short")
        # 计算权重
        # 因子与均值作差
        value = self.factor.apply(lambda x: x - x.mean(), axis=1)
        # 用value计算long和short权重
        long = value[value > 0]
        long_weight = long.apply(lambda x: x / x.sum(), axis=1).fillna(0)
        short = value[value < 0]
        short_weight = short.apply(lambda x: x / x.sum(), axis=1).fillna(0)

        long_weight = long_weight.copy().rolling(rolling_day).mean().iloc[rolling_day - 1:]
        short_weight = short_weight.copy().rolling(rolling_day).mean().iloc[rolling_day - 1:]
        close = self.close.copy().iloc[rolling_day - 1:]
        limit = self.limit.copy().iloc[rolling_day - 1:]
        stopping = self.stopping.copy().iloc[rolling_day - 1:]
        new_data = self.new_data.copy().iloc[rolling_day - 1:]

        # 涨跌停处理
        if progress_limit:
            long_weight = func(close, limit, stopping, new_data, long_weight)
            short_weight = func(close, limit, stopping, new_data, short_weight)

        mode = self.mode
        if mode == "close":
            data = new_data.copy().iloc[self.day:, ]
            l_wt = long_weight.copy().shift(self.day).iloc[self.day:, ]
            long_series = data.mul(l_wt).sum(axis=1)
            s_wt = short_weight.copy().shift(self.day).iloc[self.day:, ]
            short_series = data.mul(s_wt).sum(axis=1)

        if mode == "open":
            adj_factor = pd.read_csv("path to adj_factor", index_col=0)
            adj_factor.index = pd.to_datetime(adj_factor.index)
            open = pd.read_csv("path to open", index_col=0)
            open.index = pd.to_datetime(open.index)
            close = pd.read_csv("path to close", index_col=0)
            close.index = pd.to_datetime(close.index)
            adj_open = adj_factor * open
            adj_close = adj_factor * close
            ret_1 = adj_open.iloc[1:, :].sub(adj_close.shift(1).iloc[1:, :]).div(adj_open.shift(1).iloc[1:, :])
            ret_2 = adj_close.sub(adj_open).div(adj_open)
            ret_1 = ret_1.loc[long_weight.index, long_weight.columns].fillna(0)
            ret_2 = ret_2.loc[long_weight.index, long_weight.columns].fillna(0)
            long_series = (long_weight.shift(1 + self.day).iloc[self.day:, ].fillna(0).mul(
                ret_1.iloc[self.day:, ]) + long_weight.shift(
                self.day).iloc[self.day:, ].mul(ret_2.iloc[self.day:, ])).sum(axis=1)
            short_series = (short_weight.shift(1 + self.day).iloc[self.day:, ].fillna(0).mul(
                ret_1.iloc[self.day:, ]) + short_weight.shift(
                self.day).iloc[self.day:, ].mul(ret_2.iloc[self.day:, ])).sum(axis=1)

        # 计算日换手率
        weight = (long_weight + short_weight) / 2
        weight_t = (self.new_data.shift(1).iloc[1:, ] + 1).mul(weight.shift(1).iloc[1:, ])
        weight_t_1 = weight.iloc[1:, ]
        turnover = weight_t_1.sub(weight_t).abs().sum(axis=1)
        self.turnover = turnover
        # 计算long换手率
        l_weight = long_weight.copy()
        l_weight_t = (self.new_data.shift(1).iloc[1:, ] + 1).mul(l_weight.shift(1).iloc[1:, ])
        l_weight_t_1 = l_weight.iloc[1:, ]
        l_turnover = l_weight_t_1.sub(l_weight_t).abs().sum(axis=1)
        self.long_turnover = l_turnover
        # 计算交易费率
        transaction = self.turnover * rate
        self.long = long_series - transaction
        self.short = short_series + transaction
        # 计算差值
        difference_series = self.long.sub(self.short)
        self.difference = difference_series

    # 生成统计数据表
    def print_statis(self, path):
        print("生成统计数据表")
        l_s = self.difference.copy()
        year = l_s.resample('YE').mean().index.year
        statis = pd.DataFrame(np.zeros(shape=[year.size, 6]), index=year,
                              columns=['年化收益率', '年化波动率', '夏普比率', '最大回撤率',
                                       'Long数量', '换手率'])
        statis.index = statis.index.set_names(['long - short'])
        statis['年化收益率'] = list(l_s.resample('YE').mean() * 245)
        statis['年化波动率'] = list(l_s.resample('YE').std() * (245 ** 0.5))
        statis['夏普比率'] = statis['年化收益率'] / statis['年化波动率']
        draw_down = pd.concat([(1 + l_s).cumprod(), (1 + l_s).cumprod().expanding().max()], axis=1)
        draw_down.columns = ["current", "max"]
        draw_down['rate'] = (draw_down["max"] - draw_down["current"]) / draw_down["max"]
        statis['最大回撤率'] = list(draw_down['rate'].resample('YE').max())
        long = self.factor.sub(self.factor.mean(axis=1), axis=0) > 0
        statis['Long数量'] = list(long.sum(axis=1).resample('YE').mean())
        statis['换手率'] = list(self.turnover.resample('YE').mean() * 245)
        # ic = self.factor.shift(1).iloc[1:, :].corrwith(self.new_data.iloc[1:, :], axis=1)
        # ir = ic.resample('YE').mean() / ic.resample('YE').std()
        # statis['年化IR'] = list(ir)
        total = [l_s.mean() * 245, l_s.std() * (245 ** 0.5), (l_s.mean() * 245) / (l_s.std() * (245 ** 0.5)),
                 draw_down['rate'].max(), long.sum(axis=1).mean(), self.turnover.mean() * 245]
        statis.loc['total'] = total
        statis.to_excel(path)

    # 绘制long-short曲线
    def draw_figure(self, path):
        print("生成long-short图像")
        # 将计算结果作累计求和进行统计
        l = self.long.cumsum()
        s = self.short.cumsum()
        d = self.difference.cumsum()
        # 绘制折线图
        # 创建画布
        plt.figure(figsize=(20, 8), dpi=100)
        plt.plot(self.new_data.iloc[self.day:, ].index, l, color="r", linestyle="-", label="long")
        plt.plot(self.new_data.iloc[self.day:, ].index, s, color="b", linestyle="-", label="short")
        plt.plot(self.new_data.iloc[self.day:, ].index, d, color="g", linestyle="-", label="difference")
        # 显示图例
        plt.legend(loc="upper left")
        # 添加网格(网格效果，透明度)
        plt.grid(linestyle="--", alpha=0.5)
        # 添加坐标轴标题
        plt.xlabel("date")
        plt.ylabel("cumsum")
        # 保存图像
        plt.savefig(path)
        # 显示图像
        plt.show()

    # 读取股票close和涨跌停数据
    def read_close(self, path_close, path_limit, path_stopping):
        print("读取股票close和涨跌停数据")
        # close
        self.close = pd.read_csv(path_close, index_col=0, low_memory=False)
        self.close.index = pd.to_datetime(self.close.index)
        self.close.index.name = "date"
        # close标准化
        self.close = self.close.loc[list(set(self.factor.index).intersection(set(self.close.index))), list(
            set(self.factor.columns).intersection(set(self.close.columns)))]
        self.close = self.close.sort_index(ascending=True)
        self.close = self.close.sort_index(axis=1, ascending=True)
        # limit
        self.limit = pd.read_csv(path_limit, index_col=0, low_memory=False)
        self.limit.index = pd.to_datetime(self.limit.index)
        self.limit.index.name = "date"
        # limit标准化
        self.limit = self.limit.loc[list(set(self.factor.index).intersection(set(self.limit.index))), list(
            set(self.factor.columns).intersection(set(self.limit.columns)))]
        self.limit = self.limit.sort_index(ascending=True)
        self.limit = self.limit.sort_index(axis=1, ascending=True)
        # stopping
        self.stopping = pd.read_csv(path_stopping, index_col=0, low_memory=False)
        self.stopping.index = pd.to_datetime(self.stopping.index)
        self.stopping.index.name = "date"
        # stopping标准化
        self.stopping = self.stopping.loc[list(set(self.factor.index).intersection(set(self.stopping.index))), list(
            set(self.factor.columns).intersection(set(self.stopping.columns)))]
        self.stopping = self.stopping.sort_index(ascending=True)
        self.stopping = self.stopping.sort_index(axis=1, ascending=True)

    # 分组回策
    def group_decision(self):
        print("分组回策")
        factor = self.factor.copy()
        # 增加微小差异来消除重复值
        factor = factor.apply(add_epsilon_to_duplicates, axis=1, epsilon=0.00001)
        wt_1 = factor.copy()
        wt_2 = factor.copy()
        wt_3 = factor.copy()
        wt_4 = factor.copy()
        wt_5 = factor.copy()
        # wt_6 = factor.copy()
        # wt_7 = factor.copy()
        # wt_8 = factor.copy()
        # wt_9 = factor.copy()
        # wt_10 = factor.copy()
        for i in tqdm(range(factor.shape[0])):
            group = pd.get_dummies(pd.qcut(factor.iloc[i], 5)).astype(int)

            # # 与区间左端点作差，按比例赋权
            # left = [interval.left for interval in pd.Categorical(pd.qcut(factor.iloc[i], 5)).categories]
            # left = group.mul(left)
            # weight = group.mul(factor.iloc[i].T, axis=0).fillna(0)
            # weight = weight.sub(left)
            # weight = weight.div(weight.sum())

            # 等权赋权
            weight = group.fillna(0)
            weight = weight.div(weight.sum())

            wt_1.iloc[i] = weight.iloc[:, 0]
            wt_2.iloc[i] = weight.iloc[:, 1]
            wt_3.iloc[i] = weight.iloc[:, 2]
            wt_4.iloc[i] = weight.iloc[:, 3]
            wt_5.iloc[i] = weight.iloc[:, 4]
            # wt_6.iloc[i] = weight.iloc[:, 5]
            # wt_7.iloc[i] = weight.iloc[:, 6]
            # wt_8.iloc[i] = weight.iloc[:, 7]
            # wt_9.iloc[i] = weight.iloc[:, 8]
            # wt_10.iloc[i] = weight.iloc[:, 9]
        self.group_weight.append(wt_1)
        self.group_weight.append(wt_2)
        self.group_weight.append(wt_3)
        self.group_weight.append(wt_4)
        self.group_weight.append(wt_5)
        # self.group_weight.append(wt_6)
        # self.group_weight.append(wt_7)
        # self.group_weight.append(wt_8)
        # self.group_weight.append(wt_9)
        # self.group_weight.append(wt_10)

    # 涨跌停处理并绘制曲线图
    def limit_stopping(self, path, func1, func2, progress_limit=False, cal_transact=False):
        # 涨跌停处理
        if progress_limit:
            # 遍历各分组的权重表
            for j in range(len(self.group_weight)):
                print(f"第{j + 1}组")
                wt = self.group_weight[j].copy()
                wt = func1(self.close, self.limit, self.stopping, self.new_data, wt)
                self.group_weight[j] = wt
        # 绘制折线图
        print("绘制折线图")
        wt_1 = self.group_weight[0].copy().shift(self.day).iloc[self.day:, ]
        wt_2 = self.group_weight[1].copy().shift(self.day).iloc[self.day:, ]
        wt_3 = self.group_weight[2].copy().shift(self.day).iloc[self.day:, ]
        wt_4 = self.group_weight[3].copy().shift(self.day).iloc[self.day:, ]
        wt_5 = self.group_weight[4].copy().shift(self.day).iloc[self.day:, ]
        # wt_6 = self.group_weight[5].copy().shift(self.day).iloc[self.day:, ]
        # wt_7 = self.group_weight[6].copy().shift(self.day).iloc[self.day:, ]
        # wt_8 = self.group_weight[7].copy().shift(self.day).iloc[self.day:, ]
        # wt_9 = self.group_weight[8].copy().shift(self.day).iloc[self.day:, ]
        # wt_10 = self.group_weight[9].copy().shift(self.day).iloc[self.day:, ]
        data = self.new_data.copy().iloc[self.day:, ]
        f1 = data.mul(wt_1).sum(axis=1)
        f2 = data.mul(wt_2).sum(axis=1)
        f3 = data.mul(wt_3).sum(axis=1)
        f4 = data.mul(wt_4).sum(axis=1)
        f5 = data.mul(wt_5).sum(axis=1)
        # f6 = data.mul(wt_6).sum(axis=1)
        # f7 = data.mul(wt_7).sum(axis=1)
        # f8 = data.mul(wt_8).sum(axis=1)
        # f9 = data.mul(wt_9).sum(axis=1)
        # f10 = data.mul(wt_10).sum(axis=1)

        # 考虑交易费率
        if cal_transact:
            f1 = f1 - func2(wt_1, self.new_data)
            f2 = f2 - func2(wt_2, self.new_data)
            f3 = f3 - func2(wt_3, self.new_data)
            f4 = f4 - func2(wt_4, self.new_data)
            f5 = f5 - func2(wt_5, self.new_data)
            # f6 = f6 - func2(wt_6, self.new_data)
            # f7 = f7 - func2(wt_7, self.new_data)
            # f8 = f8 - func2(wt_8, self.new_data)
            # f9 = f9 - func2(wt_9, self.new_data)
            # f10 = f10 - func2(wt_10, self.new_data)

        self.group_data.append(f1)
        self.group_data.append(f2)
        self.group_data.append(f3)
        self.group_data.append(f4)
        self.group_data.append(f5)
        # self.group_data.append(f6)
        # self.group_data.append(f7)
        # self.group_data.append(f8)
        # self.group_data.append(f9)
        # self.group_data.append(f10)

        f1 = f1.cumsum()
        f2 = f2.cumsum()
        f3 = f3.cumsum()
        f4 = f4.cumsum()
        f5 = f5.cumsum()
        # f6 = f6.cumsum()
        # f7 = f7.cumsum()
        # f8 = f8.cumsum()
        # f9 = f9.cumsum()
        # f10 = f10.cumsum()
        d = f5.sub(f1)
        # 创建画布
        plt.figure(figsize=(20, 8), dpi=100)
        plt.plot(f1.index, f1, linestyle="-", label="group_1")
        plt.plot(f2.index, f2, linestyle="-", label="group_2")
        plt.plot(f3.index, f3, linestyle="-", label="group_3")
        plt.plot(f4.index, f4, linestyle="-", label="group_4")
        plt.plot(f5.index, f5, linestyle="-", label="group_5")
        # plt.plot(f6.index, f6, linestyle="-", label="group_6")
        # plt.plot(f7.index, f7, linestyle="-", label="group_7")
        # plt.plot(f8.index, f8, linestyle="-", label="group_8")
        # plt.plot(f9.index, f9, linestyle="-", label="group_9")
        # plt.plot(f10.index, f10, linestyle="-", label="group_10")
        plt.plot(d.index, d, color='k', linestyle="-", label="group_5 - group_1")
        # 显示图例
        plt.legend(loc="upper left")
        # 添加网格(网格效果，透明度)
        plt.grid(linestyle="--", alpha=0.5)
        # 添加坐标轴标题
        plt.xlabel("date")
        plt.ylabel("cumsum")
        # 保存图像
        plt.savefig(path)
        # 显示图像
        plt.show()

    # 输出分组统计数据表
    def group_statis(self, path1, path2):
        print("输出分组统计数据表")
        # 按年份统计
        l_s = self.group_data[4] - self.group_data[0]
        year = l_s.resample('YE').mean().index.year
        statis = pd.DataFrame(np.zeros(shape=[year.size, 5]), index=year,
                              columns=['年化收益率', '年化波动率', '夏普比率', '最大回撤率',
                                       'Top_IC'])
        statis.index = statis.index.set_names(['top - bottom'])
        statis['年化收益率'] = list(l_s.resample('YE').mean() * 245)
        statis['年化波动率'] = list(l_s.resample('YE').std() * (245 ** 0.5))
        statis['夏普比率'] = statis['年化收益率'] / statis['年化波动率']
        draw_down = pd.concat([(1 + l_s).cumprod(), (1 + l_s).cumprod().expanding().max()], axis=1)
        draw_down.columns = ["current", "max"]
        draw_down['rate'] = (draw_down["max"] - draw_down["current"]) / draw_down["max"]
        statis['最大回撤率'] = list(draw_down['rate'].resample('YE').max())
        top = self.group_weight[4].copy().iloc[:-1]
        factor = self.factor.copy().iloc[:-1]
        ret = self.new_data.copy().shift(-1).iloc[:-1]
        top_ic_lt = pd.Series(index=top.index)
        for i in tqdm(top.index):
            x = factor.loc[i][top.loc[i] != 0] - np.nanmean(factor.loc[i])
            y = ret.loc[i][top.loc[i] != 0] - np.nanmean(ret.loc[i])
            top_ic = np.nanmean(x * y) / factor.loc[i].std() / ret.loc[i].std()
            top_ic_lt.loc[i] = top_ic
        statis['Top_IC'] = list(top_ic_lt.resample('YE').mean())
        total = [l_s.mean() * 245, l_s.std() * (245 ** 0.5), (l_s.mean() * 245) / (l_s.std() * (245 ** 0.5)),
                 draw_down['rate'].max(), top_ic_lt.mean()]
        statis.loc['total'] = total
        statis.to_excel(path1)

        # 按分组统计
        f1 = self.group_data[0].copy()
        f2 = self.group_data[1].copy()
        f3 = self.group_data[2].copy()
        f4 = self.group_data[3].copy()
        f5 = self.group_data[4].copy()
        statis = pd.DataFrame(np.zeros(shape=[5, 3]),
                              index=['group_1', 'group_2', 'group_3', 'group_4', 'group_5'],
                              columns=['年化收益率', '年化波动率', '夏普比率'])
        statis.loc['group_1'] = [f1.mean() * 245, f1.std() * (245 ** 0.5),
                                 (f1.mean() * 245) / (f1.std() * (245 ** 0.5))]
        statis.loc['group_2'] = [f2.mean() * 245, f2.std() * (245 ** 0.5),
                                 (f2.mean() * 245) / (f2.std() * (245 ** 0.5))]
        statis.loc['group_3'] = [f3.mean() * 245, f3.std() * (245 ** 0.5),
                                 (f3.mean() * 245) / (f3.std() * (245 ** 0.5))]
        statis.loc['group_4'] = [f4.mean() * 245, f4.std() * (245 ** 0.5),
                                 (f4.mean() * 245) / (f4.std() * (245 ** 0.5))]
        statis.loc['group_5'] = [f5.mean() * 245, f5.std() * (245 ** 0.5),
                                 (f5.mean() * 245) / (f5.std() * (245 ** 0.5))]
        statis.index = statis.index.set_names(['group'])
        statis.to_excel(path2)

    # 和指数对比
    def index_contrast(self, path1, path2, func1, func2, top_path=None, filt_=True, a=0.1, q=0.2, rolling_day=1,
                       progress_limit=True, index=False):
        print("与指数对比")

        if top_path is not None:
            top = pd.read_csv(top_path, index_col=0)
            top.index = pd.to_datetime(top.index)
        else:
            factor = self.factor_.copy()
            factor_rank = factor.rank(axis=1, pct=True)
            p = 1 - q
            filt = factor_rank >= p
            factor_ = factor.copy()
            factor_[filt] = 1
            factor_[~filt] = 0

            top = factor_.apply(lambda x: x / x.sum(), axis=1)

        # 筛选weight
        if filt_:
            turnover = pd.read_csv(
                "path to turnover",
                index_col=0)
            turnover.index = pd.to_datetime(turnover.index)
            turnover = turnover.loc[top.index, top.columns]
            std = pd.read_csv(
                "path to std",
                index_col=0)
            std.index = pd.to_datetime(std.index)
            std = std.loc[top.index, top.columns]

            turnover_rank = turnover.rank(axis=1, pct=True)
            std_rank = std.rank(axis=1, pct=True)
            a = 1 - a
            filt = (turnover_rank >= a) | (std_rank >= a)
            top[filt] = 0
            top = top.apply(lambda x: x / x.sum(), axis=1)

        # 读取数据
        sh_300 = pd.read_csv("path to sh_300",
                             index_col=0,
                             low_memory=False)
        sh_300.index = pd.to_datetime(sh_300.index, format='%Y%m%d').strftime('%Y-%m-%d')
        sh_300.index = pd.to_datetime(sh_300.index)
        sh_300.index.name = "date"

        sh_852 = pd.read_csv("path to sh_852",
                             index_col=0,
                             low_memory=False)
        sh_852.index = pd.to_datetime(sh_852.index, format='%Y%m%d').strftime('%Y-%m-%d')
        sh_852.index = pd.to_datetime(sh_852.index)
        sh_852.index.name = "date"

        sh_905 = pd.read_csv("path to  sh_905",
                             index_col=0,
                             low_memory=False)
        sh_905.index = pd.to_datetime(sh_905.index, format='%Y%m%d').strftime('%Y-%m-%d')
        sh_905.index = pd.to_datetime(sh_905.index)
        sh_905.index.name = "date"

        csi_932 = pd.read_csv("path to csi_932",
                              index_col=0,
                              low_memory=False)
        csi_932.index = pd.to_datetime(csi_932.index, format='%Y%m%d').strftime('%Y-%m-%d')
        csi_932.index = pd.to_datetime(csi_932.index)
        csi_932.index.name = "date"

        temp = top.copy().rolling(rolling_day).mean().iloc[rolling_day - 1:]
        close = self.close_.copy().iloc[rolling_day - 1:]
        limit = self.limit_.copy().iloc[rolling_day - 1:]
        stopping = self.stopping_.copy().iloc[rolling_day - 1:]
        new_data = self.new_data_.copy().iloc[rolling_day - 1:]

        # 考虑涨跌停
        if progress_limit:
            temp = func1(close, limit, stopping, new_data, temp)

        # 计算换手率
        weight_t = (self.new_data_.shift(1).iloc[1:, ] + 1).mul(temp.shift(1).iloc[1:, ])
        weight_t_1 = temp.iloc[1:, ]
        turnover = weight_t_1.sub(weight_t).abs().sum(axis=1)
        self.top_turnover = turnover

        mode = self.mode
        if mode == "close":
            wt = temp.copy().shift(self.day).iloc[self.day:, ]
            data = new_data.copy().iloc[self.day:, ]
            f = data.mul(wt).sum(axis=1)

        if mode == "open":
            adj_factor = pd.read_csv("path to adj_factor", index_col=0)
            adj_factor.index = pd.to_datetime(adj_factor.index)
            open = pd.read_csv("path to open", index_col=0)
            open.index = pd.to_datetime(open.index)
            close = pd.read_csv("path to close", index_col=0)
            close.index = pd.to_datetime(close.index)
            adj_open = adj_factor * open
            adj_close = adj_factor * close
            ret_1 = adj_open.iloc[1:, :].sub(adj_close.shift(1).iloc[1:, :]).div(adj_open.shift(1).iloc[1:, :])
            ret_2 = adj_close.sub(adj_open).div(adj_open)
            ret_1 = ret_1.loc[temp.index, temp.columns].fillna(0)
            ret_2 = ret_2.loc[temp.index, temp.columns].fillna(0)
            f = (temp.shift(1 + self.day).iloc[self.day:, ].fillna(0).mul(ret_1.iloc[self.day:, ]) + temp.shift(
                self.day).iloc[self.day:, ].mul(ret_2.iloc[self.day:, ])).sum(axis=1)

        # 考虑交易费用
        f = f - func2(temp, self.new_data_)

        # 标准化
        i_l = list(set(f.index).intersection(set(sh_300.index)).intersection(set(sh_852.index))
                   .intersection(set(sh_905.index)).intersection(set(csi_932.index)).intersection(set(self.long_.index))
                   .intersection(set(self.short_.index)).intersection(set(self.difference_.index)))

        sh_300 = sh_300.loc[i_l]
        sh_300 = sh_300.sort_index(ascending=True)

        sh_852 = sh_852.loc[i_l]
        sh_852 = sh_852.sort_index(ascending=True)

        sh_905 = sh_905.loc[i_l]
        sh_905 = sh_905.sort_index(ascending=True)

        csi_932 = csi_932.loc[i_l]
        csi_932 = csi_932.sort_index(ascending=True)

        f_300 = sh_300.loc[:, 'ret']
        f_852 = sh_852.loc[:, 'ret']
        f_905 = sh_905.loc[:, 'ret']
        f_932 = csi_932.loc[:, 'ret']

        f5 = f.loc[i_l].sort_index(ascending=True)
        l = self.long_.loc[i_l].sort_index(ascending=True)
        s = self.short_.loc[i_l].sort_index(ascending=True)
        d = self.difference_.loc[i_l].sort_index(ascending=True)

        # 生成long_short时间序列表
        time_series = pd.DataFrame(index=l.index,
                                   columns=['long', 'short', 'long-short', 'long-HS300', 'long-ZZ500', 'long-ZZ1000',
                                            'long-ZZ2000', 'top', 'top-HS300', 'top-ZZ500', 'top-ZZ1000', 'top-ZZ2000'])
        time_series['long'] = l
        time_series['short'] = s
        time_series['long-short'] = d
        time_series['long-HS300'] = l - f_300
        time_series['long-ZZ500'] = l - f_905
        time_series['long-ZZ1000'] = l - f_852
        time_series['long-ZZ2000'] = l - f_932
        time_series['top'] = f5
        time_series['top-HS300'] = f5 - f_300
        time_series['top-ZZ500'] = f5 - f_905
        time_series['top-ZZ1000'] = f5 - f_852
        time_series['top-ZZ2000'] = f5 - f_932
        self.time_series = time_series
        time_series.to_csv(path1)

        # 绘制图像
        f_300 = f_300.cumsum()
        f_852 = f_852.cumsum()
        f_905 = f_905.cumsum()
        f_932 = f_932.cumsum()
        f5 = f5.cumsum()

        d_300 = f5.sub(f_300)
        d_852 = f5.sub(f_852)
        d_905 = f5.sub(f_905)
        d_932 = f5.sub(f_932)

        # 创建画布
        figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 8), dpi=80)
        axes[0][0].plot(f5.index, f_300, linestyle="-", label="000300.SH")
        axes[0][0].plot(f5.index, f5, linestyle="-", label="top")
        axes[0][0].plot(f5.index, d_300, linestyle="-", label="top - 000300.SH")

        axes[0][1].plot(f5.index, f_852, linestyle="-", label="000852.SH")
        axes[0][1].plot(f5.index, f5, linestyle="-", label="top")
        axes[0][1].plot(f5.index, d_852, linestyle="-", label="top - 000852.SH")

        axes[1][0].plot(f5.index, f_905, linestyle="-", label="000905.SH")
        axes[1][0].plot(f5.index, f5, linestyle="-", label="top")
        axes[1][0].plot(f5.index, d_905, linestyle="-", label="top - 000905.SH")

        axes[1][1].plot(f5.index, f_932, linestyle="-", label="932000.CSI")
        axes[1][1].plot(f5.index, f5, linestyle="-", label="top")
        axes[1][1].plot(f5.index, d_932, linestyle="-", label="top - 932000.CSI")
        # 添加网格(网格效果，透明度)
        plt.grid(linestyle="--", alpha=0.5)
        # 添加坐标轴标题
        plt.xlabel("date")
        plt.ylabel("cumsum")

        # 显示图例
        axes[0][0].legend(loc="lower right")
        axes[0][1].legend(loc="lower right")
        axes[1][0].legend(loc="lower right")
        axes[1][1].legend(loc="lower right")
        # 保存图像
        plt.savefig(path2)
        # 显示图像
        plt.show()

    # 输出时序统计量
    def series_statis(self, path):
        print("输出统计量")
        t_s = self.time_series.copy()
        l_3 = cal_statis(t_s['long-HS300'], self.long_turnover_)
        l_5 = cal_statis(t_s['long-ZZ500'], self.long_turnover_)
        l_10 = cal_statis(t_s['long-ZZ1000'], self.long_turnover_)
        l_20 = cal_statis(t_s['long-ZZ2000'], self.long_turnover_)
        t_3 = cal_statis(t_s['top-HS300'], self.top_turnover)
        t_5 = cal_statis(t_s['top-ZZ500'], self.top_turnover)
        t_10 = cal_statis(t_s['top-ZZ1000'], self.top_turnover)
        t_20 = cal_statis(t_s['top-ZZ2000'], self.top_turnover)
        with pd.ExcelWriter(path) as writer:
            l_3.to_excel(writer, sheet_name='long-HS300')
            l_5.to_excel(writer, sheet_name='long-ZZ500')
            l_10.to_excel(writer, sheet_name='long-ZZ1000')
            l_20.to_excel(writer, sheet_name='long-ZZ2000')
            t_3.to_excel(writer, sheet_name='top-HS300')
            t_5.to_excel(writer, sheet_name='top-ZZ500')
            t_10.to_excel(writer, sheet_name='top-ZZ1000')
            t_20.to_excel(writer, sheet_name='top-ZZ2000')

    def top_test(self, func1, func2, path):
        print("top划分测试")
        q_list = [0.01 * num for num in range(5, 21)]
        statis = pd.DataFrame(index=q_list, columns=["total_夏普", "2024_最大回撤"])
        sh_905 = pd.read_csv("path to sh_905",
                             index_col=0,
                             low_memory=False)
        sh_905.index = pd.to_datetime(sh_905.index, format='%Y%m%d').strftime('%Y-%m-%d')
        sh_905.index = pd.to_datetime(sh_905.index)
        sh_905.index.name = "date"
        factor = self.factor_.copy()
        factor_rank = factor.rank(axis=1, pct=True)

        for a in q_list:
            print(f"top_test : q = {a}")
            q = 1 - a
            filt = factor_rank >= q
            factor_ = factor.copy()
            factor_[filt] = 1
            factor_[~filt] = 0
            top = factor_.apply(lambda x: x / x.sum(), axis=1)
            top = func1(self.close_, self.limit_, self.stopping_, self.new_data_, top)
            wt = top.copy().shift(self.day).iloc[self.day:, ]
            data = self.new_data_.copy().iloc[self.day:, ]
            f = data.mul(wt).sum(axis=1)
            f = f - func2(wt, self.new_data_)
            i_l = list(set(f.index).intersection(set(sh_905.index)))
            sh_905 = sh_905.loc[i_l]
            sh_905 = sh_905.sort_index(ascending=True)
            f_905 = sh_905.loc[:, 'ret'].iloc[self.day:]
            f = f.loc[i_l].sort_index(ascending=True).iloc[self.day:]
            sr = f - f_905
            sharp = (sr.mean() * 245) / (sr.std() * (245 ** 0.5))
            draw_down = pd.concat([(1 + sr).cumprod(), (1 + sr).cumprod().expanding().max()], axis=1)
            draw_down.columns = ["current", "max"]
            draw_down['rate'] = (draw_down["max"] - draw_down["current"]) / draw_down["max"]
            max_down = draw_down['rate'].loc["2024-01-02":].max()
            statis.loc[a] = [sharp, max_down]
        statis.to_excel(path)
