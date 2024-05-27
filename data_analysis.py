import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from numpy import *
from numpy.linalg import multi_dot

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (10, 5)


# # 函数库

# In[19]:
def extract_excel(url='path to 模板'):
    df = pd.read_excel(url, parse_dates=True)
    df.set_index('日期', inplace=True)
    return df


# 作图函数
def pic_netvalue(sr_list, vlines=None, title=None):
    sr_cummax = sr_list[0].cummax()
    maxDDCum = (sr_cummax - sr_list[0]).cummax().fillna(method='ffill')

    fig, ax1 = plt.subplots()

    ax1.set_ylabel('净值', color='tab:red')
    for sr in sr_list:
        sr = sr.dropna()
        ax1.plot(list(sr.index), sr.values, label=sr.name)
    if vlines:
        ax1.axvline(x=vlines, color='red', linestyle='--', linewidth=0.5)

    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()

    ax2.set_ylabel('最大回撤', color='tab:blue')
    ax2.plot(list(maxDDCum.index), maxDDCum.values, color='tab:grey', linestyle='--', label='累计回撤')
    ax2.fill_between(list(maxDDCum.index), maxDDCum.values, color='tab:grey', alpha=0.2, hatch='/', edgecolor='black')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylim(0, maxDDCum.max() * 3)

    fig.tight_layout(pad=2)
    fig.legend(loc='upper left', bbox_to_anchor=(0.08, 0.95))
    if title is None:
        plt.title(sr_list[0].name)
    else:
        plt.title(title)
    plt.show()


def pic_netvalue_no_benchmark(sr, vlines=None, section=None, figsize=None):
    sr_cummax = sr.cummax()
    maxDDCum = (sr_cummax - sr).cummax().fillna(method='ffill')

    sr = sr.fillna(method='ffill')
    if figsize is None:
        fig, ax1 = plt.subplots()
    else:
        fig, ax1 = plt.subplots(figsize=(figsize[0], figsize[1]))

    ax1.set_ylabel('净值', color='tab:red')
    ax1.plot(list(sr.index), sr.values, color='tab:red', label='净值曲线')
    if section != None:
        ax1.set_ylim(section[0], section[1])
    ax1.tick_params(axis='y', labelcolor='tab:red')
    if vlines:
        ax1.axvline(x=vlines, color='red', linestyle='--', linewidth=0.5)

    ax2 = ax1.twinx()

    ax2.set_ylabel('最大回撤', color='tab:blue')
    ax2.plot(list(sr.index), maxDDCum.values, color='tab:grey', linestyle='--', label='累计回撤')
    ax2.fill_between(list(sr.index), maxDDCum.values, color='tab:grey', alpha=0.2, hatch='/', edgecolor='black')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylim(0, maxDDCum.max() * 3)

    fig.tight_layout(pad=2)
    fig.legend(loc='upper left', bbox_to_anchor=(0.08, 0.95))
    plt.title(sr.name)
    plt.show()


def plot_bar(sr):
    plt.bar(sr.index, sr.values)
    plt.title('本周净值增长条形图')
    plt.show()


# 基金业绩计算
def fund_evaluation(sr):
    sr_return = sr.dropna().pct_change()  # 收益率序列
    roe_sr_w = round(sr.dropna().iloc[-1] / sr.dropna().iloc[-6] - 1, 4)  # 近5天收益率
    roe_sr_h = round(sr.dropna().iloc[-1] / sr.dropna().iloc[0] - 1, 4)  # 持有期收益率
    roe_sr_y = round(roe_sr_h / (sr.dropna().shape[0] - 1) * 252, 4)  # 年化收益率
    std_sr = round(sr_return.dropna().values.std() * math.sqrt(252), 4)  # 年化波动率
    sharpr_sr = round((roe_sr_y - 0) / (sr_return.dropna().values.std() * math.sqrt(252)), 2)  # 夏普比率
    net_va_sr = np.array(sr.dropna().values)  # series to np.array
    net_max_sr = np.maximum.accumulate(net_va_sr)  # 累计最大值
    net_de_sr = net_max_sr - net_va_sr  # 回撤计算
    max_draw_sr = round(net_de_sr.max(), 4) + 1e-5  # 累计最大回撤
    try:
        end_index_sr = np.argmax(net_de_sr)  # 回撤结束时点
        start_index_sr = np.argmax(net_va_sr[:end_index_sr])  # 回撤开始时点
    except:
        end_index_sr = 0
        start_index_sr = 0
    end_date_sr = str.split(str(sr.dropna().index[end_index_sr]))[0]  # 格式化 时间
    start_date_sr = str.split(str(sr.dropna().index[start_index_sr]))[0]  # 格式化时间
    calmar_ratio = round((roe_sr_y - 0) / (max_draw_sr), 2)  # 卡尔玛比率
    skewness = sr_return.skew()  # 偏度
    kurtness = sr_return.kurt()  # 峰度
    win_rate = sr_return[sr_return >= 0].shape[0] / sr_return.shape[0]  # 胜率
    return roe_sr_w, roe_sr_h, roe_sr_y, std_sr, max_draw_sr, start_date_sr, end_date_sr, sharpr_sr, calmar_ratio, skewness, kurtness, win_rate


def fund_evaluation_w(sr1, freq='W-Fri'):
    sr = sr1.dropna().resample(freq).last()
    sr_return = sr.dropna().pct_change()
    roe_sr_w = round(sr.dropna().iloc[-1] / sr.dropna().iloc[-2] - 1, 4)
    roe_sr_h = round(sr.dropna().iloc[-1] / sr1.dropna().iloc[0] - 1, 4)
    roe_sr_y = round(roe_sr_h / (sr.shape[0] - 1) * 51, 4)
    std_sr = round(sr_return.dropna().values.std() * math.sqrt(51), 4)
    sharpr_sr = round((roe_sr_y - 0.00) / (sr_return.dropna().values.std() * math.sqrt(51)), 2)
    net_va_sr = np.array(sr.dropna().values)
    net_max_sr = np.maximum.accumulate(net_va_sr)
    net_de_sr = net_max_sr - net_va_sr
    max_draw_sr = round(net_de_sr.max(), 4) + 1e-5
    try:
        end_index_sr = np.argmax(net_de_sr)
        start_index_sr = np.argmax(net_va_sr[:end_index_sr])
    except:
        end_index_sr = 0
        start_index_sr = 0
    end_date_sr = str.split(str(sr.dropna().index[end_index_sr]))[0]
    start_date_sr = str.split(str(sr.dropna().index[start_index_sr]))[0]
    calmar_ratio = round((roe_sr_y - 0) / (max_draw_sr), 2)
    skewness = sr_return.skew()
    kurtness = sr_return.kurt()
    win_rate = sr_return[sr_return >= 0].shape[0] / sr_return.shape[0]
    return roe_sr_w, roe_sr_h, roe_sr_y, std_sr, max_draw_sr, start_date_sr, end_date_sr, sharpr_sr, calmar_ratio, skewness, kurtness, win_rate


def fund_comparision(*args):
    fund_table = pd.concat([arg for arg in args], axis=0)
    fund_table = fund_table.applymap(lambda x: '{:.2%}'.format(x))
    fund_table[['夏普比率', '卡尔玛比率']] = fund_table[['夏普比率', '卡尔玛比率']].applymap(
        lambda x: (float(x.strip('%')) / 100))
    return fund_table


def getFixPeriodPerfomance(return_d, freq='M'):
    if freq == 'M':
        return_M = return_d.resample('M', label='left').sum()
        return return_M

    if freq == 'Q':
        return_Q = return_d.resample('Q', label='left').sum()
        return return_Q


# # 基金类

# In[20]:


class Fund:  # 日频和周频数据略有不同
    def __init__(self, sr, freq='W-Fri', normalize=True):  # 赋值
        if normalize:
            sr = sr.dropna()
            sr = sr / sr[0]
        self.return_d = sr.pct_change()
        self.sr = sr
        self.return_w = fund_evaluation(sr)[0]
        self.return_h = fund_evaluation(sr)[1]
        self.return_y = fund_evaluation(sr)[2]
        self.std = fund_evaluation(sr)[3]
        self.max_dd = fund_evaluation(sr)[4]
        self.dd_s_date = fund_evaluation(sr)[5]
        self.dd_e_date = fund_evaluation(sr)[6]
        self.sharp_r = fund_evaluation(sr)[7]
        self.calmar_r = fund_evaluation(sr)[8]
        self.skewness = fund_evaluation(sr)[9]
        self.kurtness = fund_evaluation(sr)[10]
        self.win_rate = fund_evaluation(sr)[11]

        self.return_w_w = fund_evaluation_w(sr, freq)[0]
        self.return_h_w = fund_evaluation_w(sr, freq)[1]
        self.return_y_w = fund_evaluation_w(sr, freq)[2]
        self.std_w = fund_evaluation_w(sr, freq)[3]
        self.max_dd_w = fund_evaluation_w(sr, freq)[4]
        self.dd_s_date_w = fund_evaluation_w(sr, freq)[5]
        self.dd_e_date_w = fund_evaluation_w(sr, freq)[6]
        self.sharp_r_w = fund_evaluation_w(sr, freq)[7]
        self.calmar_r_w = fund_evaluation_w(sr, freq)[8]
        self.skewness_w = fund_evaluation_w(sr, freq)[9]
        self.kurtness_w = fund_evaluation_w(sr, freq)[10]
        self.win_rate_w = fund_evaluation_w(sr, freq)[11]

        self.name = sr.name

        sr_cummax = sr.cummax()
        maxDDCum = (sr_cummax - sr).cummax()
        self.maxDDSr = maxDDCum

    def getFundPeriodPerfomance(self, freq):
        return_d = self.return_d.copy()
        return_d = return_d.dropna()
        return_x = getFixPeriodPerfomance(return_d, freq)
        plt.bar(return_x.index, return_x.values, width=5)
        plt.title('分阶段收益率')
        plt.show()

        return return_x

    def envaluation_series(self):  # 净值评估
        columns = ['本周收益率', '持有期收益率', '年化收益率', '年化波动率', '最大回撤', '夏普比率', '卡尔玛比率',
                   '偏度', '峰度', '胜率']
        df = pd.DataFrame(data=np.zeros((1, len(columns))), index=[self.name], columns=columns)
        df_ = pd.DataFrame(data=np.zeros((1, len(columns))), index=[self.name], columns=columns)

        df.iloc[0] = [self.return_w, self.return_h, self.return_y, self.std, self.max_dd, self.sharp_r, self.calmar_r,
                      self.skewness, self.kurtness, self.win_rate]

        df_.iloc[0] = [self.return_w_w, self.return_h_w, self.return_y_w, self.std_w, self.max_dd_w, self.sharp_r_w,
                       self.calmar_r_w,
                       self.skewness_w, self.kurtness_w, self.win_rate_w]
        return df, df_

    def evaluation_benchmark(self, benchmark, freq='W-fri'):  # 超额
        ex = (self.sr.dropna() - benchmark.dropna()) + 1
        self.ex_netvalue = ex
        ex.name = str(self.sr.name) + '超额净值'
        self.benchmark = benchmark

        self.return_w_ex = fund_evaluation(ex)[0]
        self.return_h_ex = fund_evaluation(ex)[1]
        self.return_y_ex = fund_evaluation(ex)[2]
        self.std_ex = fund_evaluation(ex)[3]
        self.max_dd_ex = fund_evaluation(ex)[4]
        self.dd_s_date_ex = fund_evaluation(ex)[5]
        self.dd_e_date_ex = fund_evaluation(ex)[6]
        self.sharp_r_ex = fund_evaluation(ex)[7]
        self.calmar_r_ex = fund_evaluation(ex)[8]
        self.skewness_ex = fund_evaluation(ex)[9]
        self.kurtness_ex = fund_evaluation(ex)[10]
        self.win_rate_ex = fund_evaluation(ex)[11]

        self.return_w_ex_ = fund_evaluation_w(ex, freq)[0]
        self.return_h_ex_ = fund_evaluation_w(ex, freq)[1]
        self.return_y_ex_ = fund_evaluation_w(ex, freq)[2]
        self.std_ex_ = fund_evaluation_w(ex, freq)[3]
        self.max_dd_ex_ = fund_evaluation_w(ex, freq)[4]
        self.dd_s_date_ex_ = fund_evaluation_w(ex, freq)[5]
        self.dd_e_date_ex_ = fund_evaluation_w(ex, freq)[6]
        self.sharp_r_ex_ = fund_evaluation_w(ex, freq)[7]
        self.calmar_r_ex_ = fund_evaluation_w(ex, freq)[8]
        self.skewness_ex_ = fund_evaluation_w(ex, freq)[9]
        self.kurtness_ex_ = fund_evaluation_w(ex, freq)[10]
        self.win_rate_ex_ = fund_evaluation_w(ex, freq)[11]
        self.benchmark_name = self.benchmark.name

        columns = ['本周收益率', '持有期收益率', '年化收益率', '年化波动率', '最大回撤', '夏普比率', '卡尔玛比率',
                   '偏度', '峰度', '胜率']
        df = pd.DataFrame(data=np.zeros((1, len(columns))), index=[self.name], columns=columns)
        df_ = pd.DataFrame(data=np.zeros((1, len(columns))), index=[self.name], columns=columns)

        df.iloc[0] = [self.return_w_ex, self.return_h_ex, self.return_y_ex, self.std_ex, self.max_dd_ex,
                      self.sharp_r_ex, self.calmar_r_ex,
                      self.skewness_ex, self.kurtness_ex, self.win_rate_ex]

        df_.iloc[0] = [self.return_w_ex_, self.return_h_ex_, self.return_y_ex_, self.std_ex_, self.max_dd_ex_,
                       self.sharp_r_ex_, self.calmar_r_ex_,
                       self.skewness_ex_, self.kurtness_ex_, self.win_rate_ex_]
        return df, df_

    def print_output(self, n=0):

        if n == 0:
            print('{}自{}开始至{}业绩如下(日频):'.format(self.sr.name, str(list(self.sr.dropna().index)[0]).split()[0],
                                                         str(list(self.sr.dropna().index)[-1]).split()[0]))
            print('本基金的本周收益率为{:.2%}'.format(self.return_w))
            print('本基金的持有期收益率为{:.2%}'.format(self.return_h))
            print('本基金的年化收益率为{:.2%}'.format(self.return_y))
            print('本基金的年化波动率为{:.2%}'.format(self.std))
            print('本基金的夏普比率为%.4f' % self.sharp_r)
            print('本基金的卡尔玛比率为%.4f' % self.calmar_r)
            print('本基金的最大回撤{:.2%}, 最大回撤区间为{}至{}'.format(self.max_dd, self.dd_s_date, self.dd_e_date))
            print('本基金的胜率是{:.2%}'.format(self.win_rate))
            print('本基金的偏度{:.4f}'.format(self.skewness))
            print('本基金的峰度{:.4f}\n'.format(self.kurtness))

        elif n == 1:
            print('{}自{}开始至{}业绩如下(周频):'.format(self.sr.name, str(list(self.sr.dropna().index)[0]).split()[0],
                                                         str(list(self.sr.dropna().index)[-1]).split()[0]))
            print('本基金的本周收益率为{:.2%}'.format(self.return_w_w))
            print('本基金的持有期收益率为{:.2%}'.format(self.return_h_w))
            print('本基金的年化收益率为{:.2%}'.format(self.return_y_w))
            print('本基金的年化波动率为{:.2%}'.format(self.std_w))
            print('本基金的夏普比率为%.4f' % self.sharp_r_w)
            print('本基金的卡尔玛比率为%.4f' % self.calmar_r_w)
            print('本基金的最大回撤{:.2%}, 最大回撤区间为{}至{}'.format(self.max_dd_w, self.dd_s_date_w,
                                                                        self.dd_e_date_w))
            print('本基金的胜率是{:.2%}'.format(self.win_rate_w))
            print('本基金的偏度{:.4f}'.format(self.skewness_w))
            print('本基金的峰度{:.4f}\n'.format(self.kurtness_w))

        elif n == 2:
            print('{}自{}开始至{}超额业绩如下(日频):'.format(self.sr.name,
                                                             str(list(self.sr.dropna().index)[0]).split()[0],
                                                             str(list(self.sr.dropna().index)[-1]).split()[0]))
            print('本基金的本周收益率为{:.2%}'.format(self.return_w_ex))
            print('本基金的持有期收益率为{:.2%}'.format(self.return_h_ex))
            print('本基金的年化收益率为{:.2%}'.format(self.return_y_ex))
            print('本基金的年化波动率为{:.2%}'.format(self.std_ex))
            print('本基金的夏普比率为%.4f' % self.sharp_r_ex)
            print('本基金的卡尔玛比率为%.4f' % self.calmar_r_ex)
            print('本基金的最大回撤{:.2%}, 最大回撤区间为{}至{}'.format(self.max_dd_ex, self.dd_s_date_ex,
                                                                        self.dd_e_date_ex))
            print('本基金的胜率是{:.2%}'.format(self.win_rate_ex))
            print('本基金的偏度{:.4f}'.format(self.skewness_ex))
            print('本基金的峰度{:.4f}\n'.format(self.kurtness_ex))

        elif n == 3:
            print('{}自{}开始至{}超额业绩如下(周频):'.format(self.sr.name,
                                                             str(list(self.sr.dropna().index)[0]).split()[0],
                                                             str(list(self.sr.dropna().index)[-1]).split()[0]))
            print('本基金的本周收益率为{:.2%}'.format(self.return_w_ex_))
            print('本基金的持有期收益率为{:.2%}'.format(self.return_h_ex_))
            print('本基金的年化收益率为{:.2%}'.format(self.return_y_ex_))
            print('本基金的年化波动率为{:.2%}'.format(self.std_ex_))
            print('本基金的夏普比率为%.4f' % self.sharp_r_ex_)
            print('本基金的卡尔玛比率为%.4f' % self.calmar_r_ex_)
            print('本基金的最大回撤{:.2%}, 最大回撤区间为{}至{}'.format(self.max_dd_ex_, self.dd_s_date_ex_,
                                                                        self.dd_e_date_ex_))
            print('本基金的胜率是{:.2%}'.format(self.win_rate_ex_))
            print('本基金的偏度{:.4f}'.format(self.skewness_ex_))
            print('本基金的峰度{:.4f}\n'.format(self.kurtness_ex_))

        else:
            pass

    def plot_(self, n=0, freq='W-Fri', vlines=None):

        if n == 0:
            pic_netvalue([self.sr.dropna(), self.benchmark.dropna()], vlines=vlines)

        if n == 1:
            pic_netvalue([self.sr.dropna().resample(freq).last(),
                          self.benchmark.dropna().resample(freq).last()], vlines=vlines)
        if n == 2:
            pic_netvalue_no_benchmark(self.ex_netvalue.dropna(), vlines=vlines)

        if n == 3:
            pic_netvalue_no_benchmark(self.ex_netvalue.dropna().resample(freq).last(), vlines=vlines)

    def plot(self, n=0, vlines=None, section=None, figsize=None):
        if n == 0:
            pic_netvalue_no_benchmark(self.sr, vlines=vlines, section=section, figsize=figsize)
        if n != 0:
            pic_netvalue_no_benchmark(self.sr.resample('W-Fri').last().dropna(), vlines=vlines)


# # 大盘类

# In[21]:


class Market_Trend:
    def __init__(self, df):
        self.df = df
        self.index = df.columns
        self.start_date = df.index[0]
        self.end_date = df.index[-1]
        self.pct_change_w = self.df.pct_change(5).iloc[-1, :]
        self.pct_change_h = self.df.iloc[-1, :] / self.df.iloc[0, :] - 1

    def plot_(self):
        sr_list = []
        for column in self.df.columns:
            sr_list.append(self.df[column] / self.df[column][0])
        pic_netvalue(sr_list)

    def print_markettrend_w(self):
        a = self.pct_change_w
        print('本周大盘情况如下：')
        for column in a.index:
            print(f'{column}的本周收益率为{a[column]:.2%}')
        print('\n')

    def print_markettrend_s(self):
        a = self.pct_change_h
        print(f'自{str(self.df.index[0])[0:10]}至{str(self.df.index[-1])[0:10]}大盘业绩如下：')
        for column in a.index:
            print(f'{column}的累计收益率为{a[column]:.2%}')
        print('\n')

    def plot_bar_w(self):
        plot_bar(self.pct_change_w)
