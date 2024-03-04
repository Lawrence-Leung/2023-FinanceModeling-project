'''
    第1题 核心模型
    2023/11/2
'''

import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy.integrate import simps
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
import math
import json
import matplotlib.pyplot as plt

#--------------------------------------------------------
# 各种函数
#数据预处理
def process_and_save(input_file_path, output_file_path):
    # 读取输入的numpy文件
    input_data = np.load(input_file_path)

    # 提取每第二个维度中的a, b, c, d (即矩阵的前4列)
    extracted_data = input_data[:, :4]

    # 计算x = average(a, b, c, d)
    # axis=1表示沿着第二个维度（列）计算平均值
    averages = np.mean(extracted_data, axis=1, keepdims=True)

    # 保存新的矩阵到文件
    np.save(output_file_path, averages)

def process_and_update(input_matrix_path, input_npy_path):
    # 读取给定的numpy矩阵
    input_matrix = np.load(input_matrix_path)

    # 确定有多少个分区
    num_sections = input_matrix.shape[0] // 241  # 假设矩阵可以被241整除

    # 从每个分区中提取第2列、第1行的数据
    x = [input_matrix[i * 241, 1] for i in range(num_sections)]

    # 转换为numpy数组并调整形状以匹配目标位置
    x_array = np.array(x).reshape(-1, 1)

    # 读取'input.npy'文件
    input_npy = np.load(input_npy_path)

    # 确保'input.npy'有足够的行来容纳新数据
    rows_required = 241 * num_sections
    if input_npy.shape[0] < rows_required:
        input_npy = np.pad(input_npy, ((0, rows_required - input_npy.shape[0]), (0, 0)), mode='constant')

    # 在指定的位置插入数据x
    for i, value in enumerate(x_array):
        start_row = 1 + 241 * i
        end_row = 241 + 241 * i
        if input_npy.shape[1] < 2:
            # 如果'input.npy'只有1列，则添加新列
            new_column = np.zeros_like(input_npy[:, 0]).reshape(-1, 1)
            input_npy = np.hstack((input_npy, new_column))
        input_npy[start_row:end_row, 1] = value  # 插入数据

    # 保存更新后的'input.npy'文件
    np.save(input_npy_path, input_npy)

def insert_integer_into_string(original_string, integer, position):
    """
    将一个整数插入到字符串的指定位置。

    :param original_string: 原始字符串
    :param integer: 需要插入的整数
    :param position: 整数应插入的位置（基于0的索引）
    :return: 修改后的字符串
    """
    # 将整数转换为字符串
    integer_str = str(integer)

    # 插入整数字符串到指定位置
    new_string = original_string[:position] + integer_str + original_string[position:]

    return new_string

def top_10_entries(input_dict, output_file):
    """
    找出字典中数值最大的10个键值对并将其写入到文本文件中。

    :param input_dict: 输入的字典，其值为数值。
    :param output_file: 输出文件的名称。
    """
    # 对字典的键值对按值进行降序排序，并取前10个
    top_10 = sorted(input_dict.items(), key=lambda x: x[1], reverse=True)[:10]

    # 将这些键值对写入到指定的文本文件
    with open(output_file, 'w') as file:
        for key, value in top_10:
            file.write(f"{key}: {value}\n")

def modify_if_constant(x):
    """
    检查时间序列是否为常数值序列。
    如果是，就在序列中的每个值上添加一些微小的随机变动。

    :param x: 时间序列，一个数值列表或NumPy数组。
    :return: 可能被修改的时间序列。
    """
    if np.all(x == x[0]):  # 检查序列中的所有值是否相等
        # 在每个值上加上小的随机扰动
        noise = np.random.normal(0, 1e-8, size=len(x))
        return x + noise
    else:
        return x

#E-G协整
def check_cointegration(x, y):
    # Engle-Granger协整检验
    eg_test = sm.tsa.coint(x, y)
    p_value = eg_test[1]
    if p_value < 0.05:
        print("协整关系被确认，p-value: ", p_value)
        return True
    else:
        print("没有协整关系，p-value: ", p_value)
        return False

def build_ecm(x, y):
    # 构建误差修正模型
    model = sm.tsa.VAR(np.column_stack((x, y)))
    results = model.fit()
    return results

#时间序列预测
def get_next_y_value(model, current_y):
    # 从模型中获得预测
    forecast = model.forecast(np.column_stack((x, y)), steps=1)
    next_y = forecast[0, 1]
    return next_y

#核心交易策略 g
def g(params, x, y, profit = 0, origin_eff_num = 1000000):
    #假设从2023.8.1至2023.10.30日每个交易日交易时间（9:30~11:30、13:00~15:00)的每一分钟时间进行ETF债券交易
    i = 0  # 循环变量
    j = 241
    [d1, d2, d3, d4] = params
    total_profit_rate = []  # 用于画图的
    total_profit_chart = []  # 用于画图的
    max_y = y[5]
    min_y = y[5]
    y_tsub1 = y[5]

    for elem in y[6:1206]: #y[0:size]:
        current_y = elem   #该ETF每分钟的二级市场价
        # 获得时间序列y中t+1时刻的值
        next_y = get_next_y_value(ecm_model, current_y)     #下一分钟的二级市场价预测
        delta_y = next_y - current_y    #delta_y
        if y_tsub1 == 0:
            y_tsub1 = y_tsub1 + 1e-9
        profit_rate = elem - y_tsub1 / y_tsub1  # 利润率
        max_y = max(max_y, elem)
        min_y = min(min_y, elem)
        y_tsub1 = elem  #现在才更新elem

        #if origin_eff_num > 0: #必须得满足任何时候有ETF持仓
        if delta_y >= abs(trade_threshold):     #折价交易
            trade_1st_market = d1 * math.log(abs(delta_y))  #买入
            trade_2nd_market = d2 * math.log(abs(delta_y))  #卖出
            if i % 240 == 239:       #最后一刻补仓/平仓！
                new1 = -(trade_1st_market / (trade_1st_market + trade_2nd_market)) * origin_eff_num
                new2 = -(trade_2nd_market / (trade_1st_market + trade_2nd_market)) * origin_eff_num
                trade_1st_market = new1
                trade_2nd_market = new2
            if trade_2nd_market <= 0.7 * numpy_1[i, 5]: #限制条件（假设单次交易数量占每分钟总交易数量的比例），进行交易
                origin_eff_num = origin_eff_num + trade_1st_market - trade_2nd_market   #持仓数量更新
                temp_money = -(trade_1st_market * x[i]) + (trade_2nd_market * y[j])    #直接的买入卖出
                temp_money = temp_money - (eff1 * trade_1st_market) - (trade_2nd_market * y[j] * eff3 * 0.01) - (
                        (trade_1st_market * x[i] + trade_2nd_market * y[j]) * (eff2 + eff4) * 0.01)
                    #考虑一级市场结算金额、二级市场交易佣金、印花税和市场冲击成本
            else:
                temp_money = 0

        elif delta_y <= -1 * abs(trade_threshold):      #溢价交易
            trade_1st_market = d3 * math.log(abs(delta_y))  #卖出
            trade_2nd_market = d4 * math.log(abs(delta_y))  #买入
            if i % 240 == 239:       #最后一刻补仓/平仓！
                new1 = (trade_1st_market / (trade_1st_market + trade_2nd_market)) * origin_eff_num
                new2 = (trade_2nd_market / (trade_1st_market + trade_2nd_market)) * origin_eff_num
                trade_1st_market = new1
                trade_2nd_market = new2
            if trade_2nd_market <= 0.7 * numpy_1[i, 5]: #限制条件（假设单次交易数量占每分钟总交易数量的比例），进行交易
                origin_eff_num = origin_eff_num - trade_1st_market + trade_2nd_market   #持仓数量更新
                temp_money = +(trade_1st_market * x[i]) - (trade_2nd_market * y[j])   #直接的买入卖出
                temp_money = temp_money - (eff1 * trade_1st_market) - (trade_2nd_market * y[j] * eff3 * 0.01) - (
                            (trade_1st_market * x[i] + trade_2nd_market * y[j]) * (eff2 + eff4) * 0.01)
                # 考虑一级市场结算金额、二级市场交易佣金、印花税和市场冲击成本
            else:
                temp_money = 0

        else:       #不进行交易
            temp_money = 0
        #else:
            temp_money = 0

        profit = profit + temp_money
        total_profit_chart.append(temp_money)     #累计均利润

        total_profit_rate.append(profit_rate)  # 利润率
        if max_y == 0:
            max_y = max_y + 1e-9
        max_drawdown = float((max_y - min_y) / max_y)
        print("max_y", max_y)
        print("min_y", min_y)
        i = i + 1   #更新一次循环变量
        if (i >= size - 241):   #T+1市场
            j = i
        else:
            j = i + 241
    return profit, total_profit_chart, total_profit_rate, max_drawdown
#params: [d1, d2, d3, d4]
#x、y：原始的数据矩阵

# 定义目标函数以最大化profit
def objective(params):
    #x, y, profit, origin_eff_num = 10, 20, 30, 40  # 这些值应该根据你的具体情况设定
    new_profit = g(params, x, y, profit, origin_eff_num)
    return -new_profit  # 由于minimize函数是为了最小化目标函数，所以我们返回-new_profit以实现最大化
#--------------------------------------------------------
# Part 1 初始化及文件输入
# 指定输入和输出的文件路径

#debug only
#numpy_1_origin = 'original_dataset/62.npy'  #原始输入数据，6列n行
#numpy_2_origin = 'dataset1/62.npy'          #输出数据，2列n行

process1_result_dict = {}
total_chart_bundled = []
total_rate_bundled = []
max_drawdown_dict = {}
#with open('process1_result.json', 'w') as json_file: #打开JSON文件
#for integer in range(1, 886 + 1):
for integer in [500, 515, 449, 508, 10, 118, 18, 447, 537, 121]:
    numpy_1_origin = insert_integer_into_string('original_dataset/.npy', integer, 17)
    print("Modified String:", numpy_1_origin)
    numpy_2_origin = insert_integer_into_string('dataset1/.npy', integer, 9)
    print("Modified String:", numpy_2_origin)

    # 读取数据，用于Part 2 协整关系验证
    numpy_1 = np.load(numpy_1_origin)
    numpy_2 = np.load(numpy_2_origin)

    x, y = numpy_2[:, 1], numpy_2[:, 0]       #这个关系不要混乱了！
    #x 是一级市场价，y是二级市场价

    x = modify_if_constant(x)   #处理常数值序列问题
    y = modify_if_constant(y)

    #描述单个ETF证券的一些交易数据。
    #numpy_1.npy是[[a, b, c, d, e, f], ... ,[..., ..., ..., ..., ..., ..., ]]，即矩阵有6列，若干行(假设为dim_2行)；
    #其中，a代表每分钟的收盘价，b代表每分钟开盘价，c代表最高价，d代表最低价，e代表每分钟的总成交额，f代表每分钟该ETF证券的成交量。每一行代表每个交易时刻（以分钟为单位）的数据。

    #numpy_2.npy是一个2列、若干行(同样是dim_2行)的矩阵，每行的数据为[g, h]，
    #其中g代表该ETF每分钟的二级市场价，h代表该ETF每分钟的一级市场价。同样地，每一行代表每个交易时刻（以分钟为单位）的数据。


    # Part 2 协整关系构建
    # 检查协整关系
    cointegrated = check_cointegration(x, y)
    # 构建误差修正模型
    ecm_model = build_ecm(x, y)

    # Part 3 基于每个股票的参数处理及预测：全部放到后面去

    # 折溢价率
    #c1 = calculate_premium_discount_rate(numpy_1, numpy_2)
    #print(f'平均折溢价率: {c1:.4f}%')

    # 跟踪误差
    #c2 = calculate_tracking_error(numpy_1, numpy_2)
    #print(f'跟踪误差: {c2:.4f}%')

    # 宏观经济指标
    #c3 = calculate_macro_economic_indicator(numpy_1, numpy_2)
    #print(f'宏观经济指标: {c3:.2f}')


    # Part 4 交易策略
    # 这几个变量都是需要被预测的，待定系数
    [d1, d2, d3, d4] = [15.993214229999998 * 1, -6.653766326666666 * 1, 5.151237160000001 * 1, 62.70280344 * 0.75]
    params = [d1, d2, d3, d4]
    trade_threshold = 1e-5  #交易阈值信号限制：0.00001元

    profit = 0      #我们自己的利润

    eff1 = 1    #ETF申购、赎回费用、过户费、证管费用、经手费、证券结算金:固定成本（元）（一级市场）
    eff2 = 0.1  #印花税率（百分比%）（所有市场）
    eff3 = 0.2  #ETF二级市场交易佣金、股票交易佣金百分比%）（二级市场）
    eff4 = 0.05 #市场冲击成本（百分比%）（所有市场）
    trade_regulation = 0.7  #假设单次交易数量占每分钟总交易数量的比例
    origin_eff_num = 1000000    #限购数量100万ETF

    size = min(len(numpy_1), len(numpy_2))

    # 进行参数估计-----------------------------------------
        # 初始猜测的d1, d2, d3, d4参数值
        #initial_guess = [1, 1, 1, 1]

        # 使用minimize函数来寻找最优参数
        #result = minimize(objective, initial_guess, method='BFGS')  # BFGS是一个常用的优化算法

        # 输出最优参数
        #optimal_params = result.x
        #print(f'Optimal parameters: {optimal_params}')

        # 现在你可以使用这些最优参数来计算最大profit
        #max_profit = -objective(optimal_params)
        #print(f'Maximum profit: {max_profit}')
    #-----------------------------------------------------

    # 参数输出

    new_profit, total_profit_chart, total_profit_rate, max_drawdown_result = g(params, x, y, profit, origin_eff_num)
    total_profit_chart = np.array(total_profit_chart)
    total_chart_bundled.append(total_profit_chart)
    total_profit_rate = np.array(total_profit_rate)
    total_rate_bundled.append(total_profit_rate)

    process1_result_dict[integer] = new_profit
    max_drawdown_dict[integer] = max_drawdown_result
    with open('process2_1comp.json', 'w') as json_file:
        json.dump(process1_result_dict, json_file, indent=4)
    with open('process1_maxdrawdown_deleted.json', 'w') as json_file:
        json.dump(max_drawdown_dict, json_file, indent=4)
    #关闭文件，退出循环

#top_10_entries(process1_result_dict, "process1_top_10.txt")

# 画图-------------------------------------------------
# 设置图表的标题、x轴和y轴的标签
'''
titles = ['511620.SH', '511650.SH', '511920.SH', '511600.SH', '511800.SH', '511830.SH', '511850.SH', '511900.SH', '511810.SH',
          '511820.SH']
x_labels = ['Trade Time/min'] * 10
y_labels = ['Unit Profit/(CNY/min$^{-1}$)'] * 10

# 创建图形
fig, axs = plt.subplots(2, 5, figsize=(15, 6))

# 遍历每个子图并绘制数据
for i, ax in enumerate(axs.flat):
    #ax.bar(range(len(total_chart_bundled[i])), total_chart_bundled[i])
    ax.plot(total_chart_bundled[i])
    ax.set_title(titles[i])
    ax.set_xlabel(x_labels[i])
    ax.set_ylabel(y_labels[i])

# 调整子图间的空间以保证标签不重叠
plt.tight_layout()
plt.show()

#-----------------------------------------

titles = ['511620.SH', '511650.SH', '511920.SH', '511600.SH', '511800.SH', '511830.SH', '511850.SH', '511900.SH', '511810.SH',
          '511820.SH']
x_labels = ['Trade Time/min'] * 10
y_labels = ['Profit Rate'] * 10
# 显示图形
plt.show()

# 创建图形
fig, axs = plt.subplots(2, 5, figsize=(15, 6))

# 遍历每个子图并绘制数据
for i, ax in enumerate(axs.flat):
    #ax.bar(range(len(total_chart_bundled[i])), total_chart_bundled[i])
    ax.plot(total_rate_bundled[i])
    ax.set_title(titles[i])
    ax.set_xlabel(x_labels[i])
    ax.set_ylabel(y_labels[i])

# 调整子图间的空间以保证标签不重叠
plt.tight_layout()

# 显示图形
plt.show()
'''