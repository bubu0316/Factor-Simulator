import pandas as pd
import Common.progress_limit_stopping
import Common.transaction_calculate
import Common.function
# 导入Simulator类
from Common.simulator import Simulator
import os


# 文件路径
file_path = "path to output dir"
# 创建目录
if not os.path.exists(file_path + "output"):
    os.mkdir(file_path + "output")
if not os.path.exists(file_path + "output/csv"):
    os.mkdir(file_path + "output/csv")
if not os.path.exists(file_path + "output/figure"):
    os.mkdir(file_path + "output/figure")

mode = "close"
day = 1

# 创建模拟器对象
s1 = Simulator()
# 读取日收益率
s1.read_data(mode=mode, day=day)
# 计算因子及处理原始数据
s1.calculate_factor(func=Common.function.cal_factor, path="path to factor", date="start date")

# 读取行业数据
s1.read_industry("path to industry")
# 对因子数据做数据清洗
s1.factor_clean()


# 读取市值数据
s1.read_mv("path to mv")
# 因子中性化
s1.factor_neutral(mv_=False, ind_=True)

# 读取股票close和涨跌停数据
s1.read_close(path_close="path to close",
              path_limit="path to limit",
              path_stopping="path to stopping",)

# 计算long,short及其差值
s1.calculate_difference(func=Common.progress_limit_stopping.p_l_s, progress_limit=True, rolling_day=5)

# 生成统计数据表
s1.print_statis(file_path + "output/csv/long_short_statis.xlsx")
# 绘制long-short曲线
s1.draw_figure(file_path + "output/figure/因子long-short对比图.png")

# 分组回策
s1.group_decision()
# 涨跌停处理并绘制曲线图
s1.limit_stopping(file_path + "output/figure/因子分组回策图.png",
                  func1=Common.progress_limit_stopping.p_l_s, func2=Common.transaction_calculate.cal_transaction,
                  progress_limit=True, cal_transact=True)
# 输出分组统计表
s1.group_statis(path1=file_path + "output/csv/group_difference_statis.xlsx",
                path2=file_path + "output/csv/group_compare_statis.xlsx")

# 指数对比
s1.index_contrast(path1=file_path + "output/csv/时间序列.csv", path2=file_path + "output/figure/指数对比.png",
                  func1=Common.progress_limit_stopping.p_l_s, func2=Common.transaction_calculate.cal_transaction,
                  filt_=True, q=0.2, rolling_day=5, progress_limit=True)
# 输出时序统计量
s1.series_statis(file_path + "output/统计量.xlsx")

# top划分测试
s1.top_test(func1=Common.progress_limit_stopping.p_l_s, func2=Common.transaction_calculate.cal_transaction,
            path=file_path + "output/top_ZZ500.xlsx")
