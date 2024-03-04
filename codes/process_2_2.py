'''
    第2题 核心模型（改进后）
    2023/11/2
'''
import numpy as np
import numpy.polynomial.polynomial as poly  #多重分形离散分析（MF-DFA）方法
from scipy.integrate import simps
from scipy.optimize import minimize_scalar
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import os
import math

#函数定义部分------------------------------------------------------
#数据预处理
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
def modify_if_constant(x):
    """
    检查时间序列是否为常数值序列。（改动过了）
    如果是，就在序列中的每个值上添加一些微小的随机变动。

    :param x: 时间序列，一个数值列表或NumPy数组。
    :return: 可能被修改的时间序列。
    """
    '''
    if np.all(x == x[0]):  # 检查序列中的所有值是否相等
        # 在每个值上加上小的随机扰动
        noise = np.random.normal(0, 1e-9, size=len(x))
        return x + noise
    else:
        return x
    '''
    noise = np.random.normal(0, 1e-9, size=len(x))
    return x + noise
def list_sum_perdata(list1, list2):
    #把两个数组按元素增加
    result = [x + y for x,y in zip(list1, list2)]
    return result
def normalize_array(arr):
    #把一个数组的数据标准化处理
    min_val = np.min(arr)
    max_val = np.max(arr)
    range_val = max_val - min_val
    if range_val != 0:
        normalized_arr = (arr - min_val) / range_val
        return normalized_arr
    else:
        return arr  # 如果数组中所有值都相同，则返回原数组
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
def extract_numbers_from_json(file_path):
    # 用于存储所有数值的列表
    numbers_list = []

    # 递归函数来遍历 JSON 数据结构
    def extract_numbers(data):
        if isinstance(data, dict):  # 如果是字典，递归它的值
            for value in data.values():
                extract_numbers(value)
        elif isinstance(data, list):  # 如果是列表，递归每个元素
            for item in data:
                extract_numbers(item)
        elif isinstance(int(data), (int, float)):  # 如果是数值，添加到列表
            numbers_list.append(data)

    # 读取 JSON 文件并将内容加载到 Python 数据结构中
    with open(file_path, 'r') as file:
        data = json.load(file)
        extract_numbers(data)

    return numbers_list
#折溢价率
def calculate_premium_discount_rate(numpy_1):
    # 提取二级市场价格和一级市场价格
    market_prices = numpy_1[:, 0]
    nav_prices = numpy_1[:, 1]

    # 替换零值以避免除以零的错误
    epsilon = 10  # 一个正数，由经验确定
    nav_prices_replaced = np.where(nav_prices == 0, epsilon, nav_prices)

    # 计算每分钟的折溢价率
    premium_discount_rates = ((market_prices - nav_prices_replaced) / nav_prices_replaced) * 100

    # 计算折溢价率的平均值
    average_premium_discount_rate = np.mean(premium_discount_rates)

    return average_premium_discount_rate
#跟踪误差
def calculate_tracking_error(numpy_1):
    # 提取二级市场价格和一级市场价格
    market_prices = numpy_1[:, 0]
    nav_prices = numpy_1[:, 1]

    # 确保没有零值，以避免除以零的错误
    epsilon = 2.71828  # 一个正数，由经验确定
    nav_prices_replaced = np.where(nav_prices == 0, epsilon, nav_prices)

    # 计算每分钟的价格差异比率
    price_diff_ratios = (market_prices - nav_prices_replaced) / nav_prices_replaced

    # 计算跟踪误差
    tracking_error = np.std(price_diff_ratios)
    return tracking_error
#宏观经济指标，内部有嵌套
def calculate_macro_economic_indicator(numpy_1, numpy_2):
    # 从numpy_1矩阵中提取数据
    closing_prices = numpy_1[:, 0]
    total_trade_value = numpy_1[:, 4]
    trade_volume = numpy_1[:, 5]

    # 计算价格动量（二阶偏导数）
    price_momentum = np.gradient(np.gradient(closing_prices))

    # 计算成交量动量（二阶偏导数）
    volume_momentum = np.gradient(np.gradient(trade_volume))

    # 计算市场流动性（每分钟的总成交额与收盘价的比值）
    market_liquidity = total_trade_value / (closing_prices + 1e-10)  # 避免除以零

    # 定义一个滑动窗口函数来计算局部平均值
    def sliding_window(arr, window_size):
        return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

    # 使用滑动窗口计算局部平均价格动量和成交量动量
    local_avg_price_momentum = sliding_window(price_momentum, 10)
    local_avg_volume_momentum = sliding_window(volume_momentum, 10)

    # 定义一个优化目标函数，该函数试图最大化价格动量和成交量动量之间的相关性
    def objective_func(window_size):
        local_avg_price_momentum = sliding_window(price_momentum, int(window_size))
        local_avg_volume_momentum = sliding_window(volume_momentum, int(window_size))
        correlation = np.corrcoef(local_avg_price_momentum, local_avg_volume_momentum)[0, 1]
        return -correlation  # 最大化相关性相当于最小化负相关性

    # 执行优化以找到最佳滑动窗口大小
    optimal_window_size = minimize_scalar(objective_func, bounds=(5, 50), method='bounded').x

    # 使用最佳滑动窗口大小重新计算局部平均价格动量和成交量动量
    local_avg_price_momentum = sliding_window(price_momentum, int(optimal_window_size))
    local_avg_volume_momentum = sliding_window(volume_momentum, int(optimal_window_size))

    # 将三个指标组合成一个复合指标
    # 我们假设每个指标的权重为1/3，可以根据实际情况调整权重
    length1 = len(local_avg_price_momentum)
    length2 = len(local_avg_volume_momentum)
    length3 = len(market_liquidity)
    min_length = min(length1, length2, length3)

    composite_indicator = (local_avg_price_momentum[0:min_length] + local_avg_volume_momentum[0:min_length] + market_liquidity[0:min_length]) / 3

    # 计算复合指标的积分作为宏观经济参数化指标
    macro_economic_indicator = simps(composite_indicator) / (10 ** 7)

    return macro_economic_indicator
# Hurst 矩阵处理函数————————————————
#多重分形离散分析（MF-DFA）方法
def MF_DFA(X, q_values=None):
    if q_values is None:
        q_values = [1, 2, 3, 4, 5, 6, 7]

    # Ensure X is a numpy array of type float64
    X = np.asarray(X, dtype=np.float64)

    N = len(X)
    # 构建累积偏差序列
    Y = np.cumsum(X - np.mean(X))

    # 分段和局部趋势拟合
    # 选择合适的段长度s，这里选择为7
    s = N // 7
    Fq = np.zeros(len(q_values))

    for i, q in enumerate(q_values):
        F2 = np.zeros(s)
        for v in range(0, N, s):
            if v + s < N:
                # 对每个段使用多项式拟合
                segment = Y[v:v + s]
                t = np.arange(len(segment))
                p = poly.Polynomial.fit(t, segment, 3)  # 使用三次多项式拟合
                fit = p(t)
                # 计算离散函数
                F2[v // s] = np.mean((segment - fit) ** 2)
        # 求解标度函数
        Fq[i] = np.mean(F2 ** (q / 2)) ** (1 / q) if q != 0 else np.exp(0.5 * np.mean(np.log(F2)))

    # 计算h(q)
    hq = np.log(Fq) / np.log(s)
    return hq
#从2个数到1*7向量的变换
def transform_to_vector(c1, c2):
    return np.array([c1, c2, c1*c2, c1**2, c2**2, np.sqrt(abs(c1)), np.sqrt(abs(c2))])
#线性缩放方法
def scale_to_range(vector, min_val, max_val):
    min_vector = np.min(vector)
    max_vector = np.max(vector)
    scaled_vector = (max_val - min_val) * (vector - min_vector) / (max_vector - min_vector) + min_val
    return scaled_vector
#聚合和线性缩放的方法
def aggregate_and_scale(vector, num_elements, seed=0):
    # 设置随机种子以保证打散操作的一致性
    np.random.seed(seed)

    # 首先随机打散向量
    np.random.shuffle(vector)

    # 将打散后的向量分组并聚合
    aggregated = [np.mean(vector[i:i + len(vector) // num_elements]) for i in
                  range(0, len(vector), len(vector) // num_elements)]

    # 确保聚合后的列表有num_elements个元素
    if len(aggregated) > num_elements:
        aggregated = aggregated[:num_elements]

    # 线性缩放到1到2之间
    min_val, max_val = 1, 2
    min_aggregated = np.min(aggregated)
    max_aggregated = np.max(aggregated)

    # 如果所有聚合值相等（防止除以0）
    if max_aggregated == min_aggregated:
        scaled = [min_val + (max_val - min_val) / 2] * num_elements  # 所有值设为中点
    else:
        scaled = [(max_val - min_val) * (val - min_aggregated) / (max_aggregated - min_aggregated) + min_val for val in
                  aggregated]

    return scaled
# 时间序列处理函数——————————————————
#基于时间偏移的预测
def exponential_decay_weight(n, decay_rate):
    """
    计算指数衰减权重。

    :param n: 时间偏移量。
    :param decay_rate: 衰减率。
    :return: 权重值。
    """
    return np.exp(-decay_rate * n)
def corrected_price_prediction(predictions, decay_rate=0.2):
    """
    根据预测值和时间偏移量计算修正后的价格预测值。
    :param predictions: 包含t+1, t+2, t+3时刻的价格预测值的数组。
    :param decay_rate: 衰减率，用于计算权重。
    :return: 修正后的t+1时刻的价格预测值。
    """
    # 确保predictions是一个一维numpy数组
    predictions = np.squeeze(np.array(predictions))
    # 计算权重
    weights = np.array([exponential_decay_weight(n, decay_rate) for n in range(1, len(predictions) + 1)])
    # 标准化权重
    normalized_weights = weights / np.sum(weights)
    # 计算加权平均值
    corrected_prediction = np.dot(predictions, normalized_weights)
    corrected_prediction = corrected_prediction + 0.0024
    return corrected_prediction

#核心交易策略 gT0
def gT0(params, x, y, Y_aggregated, profit = 0, origin_eff_num = 1000000):
    #假设从2023.8.1至2023.10.30日每个交易日交易时间（9:30~11:30、13:00~15:00)的每一分钟时间进行ETF债券交易
    i = 0  # 循环变量
    j = 0
    [d1, d2, d3, d4] = params
    total_profit_rate = []   # 用于画图的
    total_profit_chart = []  # 用于画图的
    origin_y = []
    predict_y = []
    n_input_steps = 7
    n_output_steps = 3
    max_y = y[n_input_steps - 1]
    min_y = y[n_input_steps - 1]
    e1, e2, e3, e4 = Y_aggregated   #参数处理
    #time_steps = 5 # 时间序列的时间步长
    extracted_time_series_data = []

    for i in range(1200):#len(y) - n_input_steps - n_output_steps + 1):
        if i % 2000 == 0:
            print(i)
        # 将提取的数据添加到列表中
        extracted_time_series_data.append(y[i:i + n_input_steps])
    extracted_time_series_data = np.array(extracted_time_series_data)
    extracted_new_y_data = model.predict(extracted_time_series_data, verbose=1, batch_size=1000, workers=-1)  # 批量预测

    for t in range(1200):   #len(extracted_time_series_data)):     #模拟所有周期的数据
        # 预测时间序列中t+1, t+2, t+3时刻的值
        #if t % 500 == 0:
            #print(extracted_time_series_data[t])
        #new_y_data = model.predict(extracted_time_series_data[t], batch_size=1000, verbose=0, workers=-1)
        #print(new_y_data)
        delta_y = corrected_price_prediction(extracted_new_y_data[t]) - y[t]    #修正为一个值
        #print(delta_y)
        absolute_delta_y = y[t+1] - y[t]    #delta_y

        profit_rate = (absolute_delta_y / y[t])  #利润率
        max_y = max(max_y, y[t])
        min_y = min(min_y, y[t])

        #if origin_eff_num > 0: #必须得满足任何时候ETF持仓不为负
        if delta_y >= abs(trade_threshold):     #折价交易
            trade_1st_market = e1 * d1 * math.log(abs(delta_y))  #买入
            trade_2nd_market = e2 * d2 * math.log(abs(delta_y))  #卖出
            if t % 240 == 239:       #最后一刻补仓/平仓！
                new1 = -(trade_1st_market / (trade_1st_market + trade_2nd_market)) * origin_eff_num
                new2 = -(trade_2nd_market / (trade_1st_market + trade_2nd_market)) * origin_eff_num
                trade_1st_market = new1
                trade_2nd_market = new2
            if trade_2nd_market <= 0.7 * numpy_1[t, 5]: #限制条件（假设单次交易数量占每分钟总交易数量的比例），进行交易
                origin_eff_num = origin_eff_num + trade_1st_market - trade_2nd_market   #持仓数量更新
                temp_money = -(trade_1st_market * x[t]) + (trade_2nd_market * y[j])    #直接的买入卖出
                temp_money = temp_money - (eff1 * trade_1st_market) - (trade_2nd_market * y[j] * eff3 * 0.01) - (
                        (trade_1st_market * x[t] + trade_2nd_market * y[j]) * (eff2 + eff4) * 0.01)
                    #考虑一级市场结算金额、二级市场交易佣金、印花税和市场冲击成本
            else:
                temp_money = 0

        elif delta_y <= -1 * abs(trade_threshold):      #溢价交易
            trade_1st_market = e3 * d3 * math.log(abs(delta_y))  #卖出
            trade_2nd_market = e4 * d4 * math.log(abs(delta_y))  #买入
            if t % 240 == 239:       #最后一刻补仓/平仓！
                new1 = -(trade_1st_market / (trade_1st_market + trade_2nd_market)) * origin_eff_num
                new2 = -(trade_2nd_market / (trade_1st_market + trade_2nd_market)) * origin_eff_num
                trade_1st_market = new1
                trade_2nd_market = new2
            if trade_2nd_market <= 0.7 * numpy_1[t, 5]: #限制条件（假设单次交易数量占每分钟总交易数量的比例），进行交易
                origin_eff_num = origin_eff_num - trade_1st_market + trade_2nd_market   #持仓数量更新
                temp_money = +(trade_1st_market * x[t]) - (trade_2nd_market * y[j])   #直接的买入卖出
                temp_money = temp_money - (eff1 * trade_1st_market) - (trade_2nd_market * y[j] * eff3 * 0.01) - (
                            (trade_1st_market * x[t] + trade_2nd_market * y[j]) * (eff2 + eff4) * 0.01)
                # 考虑一级市场结算金额、二级市场交易佣金、印花税和市场冲击成本
            else:
                temp_money = 0

        else:       #不进行交易
            temp_money = 0
        #else:
            temp_money = 0


        profit = profit + temp_money
        total_profit_chart.append(temp_money)     #累计利润
        total_profit_rate.append(profit_rate)   #利润率
        origin_y.append(y[t])
        predict_y.append(delta_y + y[t])    #新y值(从7至1207)
        max_drawdown = float((max_y - min_y) / max_y)
        #i = i + 1   #更新一次循环变量
        '''if (i >= size - 241):   #T+1市场
            j = i
        else:
            j = i + 241'''
        j = t   #T+0市场
        if j % 50 == 0:
            print(j, "Profit=" ,profit)
    return profit, total_profit_chart, total_profit_rate, max_drawdown, predict_y, origin_y
#params: [d1, d2, d3, d4]
#x、y：原始的数据矩阵
#实际执行部分------------------------------------------------------
#交易参数
[d1, d2, d3, d4] = [15.993214229999998, -6.653766326666666, 5.151237160000001, 62.70280344 * 0.75]
params = [d1, d2, d3, d4]
trade_threshold = 0  # 交易阈值信号限制：0.00001元
profit = 0  # 我们自己的利润
eff1 = 1  # ETF申购、赎回费用、过户费、证管费用、经手费、证券结算金:固定成本（元）（一级市场）
eff2 = 0.1  # 印花税率（百分比%）（所有市场）
eff3 = 0.2  # ETF二级市场交易佣金、股票交易佣金百分比%）（二级市场）
eff4 = 0.05  # 市场冲击成本（百分比%）（所有市场）
trade_regulation = 0.7  # 假设单次交易数量占每分钟总交易数量的比例
origin_eff_num = 1000000  # 假设初始持仓数量100万ETF

#以下是训练用————————————————————————————————————————————————
"""
#构造LSTM模型---------------
## LSTM网络的参数
np.random.seed(1)
tf.random.set_seed(1)
'''
# 构建LSTM与全连接层结合的模型
model = Sequential()
# LSTM层
model.add(LSTM(50, activation='relu', input_shape=(7,1), return_sequences=False))
# 全连接层
model.add(Dense(20, activation='relu'))  # 添加一个有20个单元的全连接层
model.add(Dense(3))  # 输出层，预测t+1, t+2, t+3时刻的价格
# 编译模型
model.compile(optimizer='adam', loss='mse')
'''
model = load_model('process2_100_lstm_model.h5')
model.summary()

#准备开始
ETF100index_json_file_path = 'process2_3_100ETFindexes.json'
integers = extract_numbers_from_json(ETF100index_json_file_path)
print(integers)
for integer in integers:
    # 从文件中读取数据 --------------
    numpy_1_origin = insert_integer_into_string('dataset1/.npy', integer, 9)  # 按照索引读取numpy数据
    print("Modified String:", numpy_1_origin)
    numpy_2_origin = insert_integer_into_string('dataset2/.npy', integer, 9)
    print("Modified String:", numpy_2_origin)
    numpy_original_origin = insert_integer_into_string('original_dataset/.npy', integer, 17)

    if not os.path.exists(numpy_2_origin):
        continue
    numpy_1 = np.load(numpy_1_origin, allow_pickle=True)
    numpy_2 = np.load(numpy_2_origin, allow_pickle=True)
    numpy_original = np.load(numpy_original_origin, allow_pickle=True)

    market2_1 = numpy_original[:, 0]  # 收盘价
    market2_2 = numpy_original[:, 1]  # 开盘价
    market2_3 = numpy_original[:, 2]  # 最高价
    market2_4 = numpy_original[:, 3]  # 最低价
    market2_1 = modify_if_constant(market2_1)
    market2_2 = modify_if_constant(market2_2)
    market2_3 = modify_if_constant(market2_3)
    market2_4 = modify_if_constant(market2_4)

    market2_5 = numpy_original[:, 4]  # 每分钟成交额
    market2_6 = numpy_original[:, 5]  # 每分钟成交量

    market1_1 = numpy_2[:, 0]  # 复权单位净值
    market1_2 = numpy_2[:, 1]  # 贴水
    market1_3 = numpy_2[:, 2]  # 贴水率
    market1_4 = numpy_2[:, 3]  # 增长率

    # 计算MF-DFA --------------
    hurst_matrix = []
    sum_matrix = []
    sum_matrix = list_sum_perdata(MF_DFA(market2_1), MF_DFA(market2_2))  # 二级市场四个数据叠加
    sum_matrix = list_sum_perdata(sum_matrix, MF_DFA(market2_3))
    sum_matrix = list_sum_perdata(sum_matrix, MF_DFA(market2_4))
    sum_matrix = np.array(sum_matrix) * 0.25
    hurst_matrix.append(sum_matrix)

    sum_matrix = MF_DFA(market2_5)  # 每分钟成交额
    hurst_matrix.append(sum_matrix)
    sum_matrix = MF_DFA(market2_6)  # 每分钟成交量
    hurst_matrix.append(sum_matrix)

    sum_matrix = MF_DFA(market1_1)  # 每分钟成交额
    hurst_matrix.append(sum_matrix)
    sum_matrix = MF_DFA(market1_2)  # 每分钟成交量
    hurst_matrix.append(sum_matrix)
    sum_matrix = MF_DFA(market1_3)  # 每分钟成交额
    hurst_matrix.append(sum_matrix)
    sum_matrix = MF_DFA(market1_4)  # 每分钟成交量
    hurst_matrix.append(sum_matrix)
    hurst_matrix = np.array(hurst_matrix)

    # 计算折溢价率和跟踪误差 --------------
    c1 = calculate_premium_discount_rate(numpy_1)
    print(f'平均折溢价率: {c1:.4f}%')
    c2 = calculate_tracking_error(numpy_1)
    print(f'跟踪误差: {c2:.4f}%')
    # 修正H(q)矩阵 --------------

    # 时间序列及处理，用作训练 --------------
    y = numpy_1[:, 0]  # 这个关系不要混乱了！我们最后要取得的是y
    x = numpy_1[:, 1]

    # 初始化空列表来存储提取的时间序列数据和目标值
    extracted_time_series_data = [] #输入的值
    extracted_target_values = []
    # 遍历数组，假设t从第7个时间点开始，到倒数第3个时间点结束
    # 这样可以确保我们能够提取t-6到t和t+1到t+2的数据
    n_input_steps = 7
    n_output_steps = 3
    for i in range(len(y) - n_input_steps - n_output_steps + 1):
        if i % 2000 == 0:
            print(i)
        # 将提取的数据添加到列表中
        extracted_time_series_data.append(y[i:i + n_input_steps])
        extracted_target_values.append(y[i + n_input_steps:i + n_input_steps + n_output_steps])
    extracted_target_values = np.array(extracted_target_values)
    extracted_time_series_data = np.array(extracted_time_series_data)
    history = model.fit(extracted_time_series_data, extracted_target_values, epochs=180, validation_split=0.2, verbose=1)
    print('Model Fitted!')
    model.save('process2_100_lstm_model2.h5')  # 保存
    #with open('process2_10_modelhistory.json', 'w') as json_file:
    #   json.dump(history, json_file, indent=4)  # 写入到文件中去
model.save('process2_3_lstm_model2.h5')  #保存
"""

#以下是推理用————————————————————————————————————————————————
model = load_model('process2_100_lstm_model.h5')
model.summary()

process2_profit_dict = {}
max_drawdown_dict = {}
total_chart_bundled = []
total_rate_bundled = []
predicted_y_bundled = []
origin_y_bundled = []

#遍历每一个文件-----------------------------------------------------
ETF100index_json_file_path = 'process2_3_100ETFindexes.json'
#integers = extract_numbers_from_json(ETF100index_json_file_path)
integers = [500, 515, 449, 508, 10, 118, 18, 447, 537, 121]

print(integers)
for integer in integers:
    # 从文件中读取数据 --------------
    numpy_1_origin = insert_integer_into_string('dataset1/.npy', integer, 9)    #按照索引读取numpy数据
    print("Modified String:", numpy_1_origin)
    numpy_2_origin = insert_integer_into_string('dataset2/.npy', integer, 9)
    print("Modified String:", numpy_2_origin)
    numpy_original_origin = insert_integer_into_string('original_dataset/.npy', integer, 17)

    if not os.path.exists(numpy_2_origin):
        continue
    numpy_1 = np.load(numpy_1_origin, allow_pickle=True)
    numpy_2 = np.load(numpy_2_origin, allow_pickle=True)
    numpy_original = np.load(numpy_original_origin, allow_pickle=True)

    market2_1 = numpy_original[:, 0]    #收盘价
    market2_2 = numpy_original[:, 1]    #开盘价
    market2_3 = numpy_original[:, 2]    #最高价
    market2_4 = numpy_original[:, 3]    #最低价
    market2_1 = modify_if_constant(market2_1)
    market2_2 = modify_if_constant(market2_2)
    market2_3 = modify_if_constant(market2_3)
    market2_4 = modify_if_constant(market2_4)

    market2_5 = numpy_original[:, 4]    #每分钟成交额
    market2_6 = numpy_original[:, 5]    #每分钟成交量

    market1_1 = numpy_2[:, 0]   #复权单位净值
    market1_2 = numpy_2[:, 1]   #贴水
    market1_3 = numpy_2[:, 2]   #贴水率
    market1_4 = numpy_2[:, 3]   #增长率

    # 计算MF-DFA --------------
    hurst_matrix = []
    sum_matrix = []
    sum_matrix = list_sum_perdata(MF_DFA(market2_1), MF_DFA(market2_2)) #二级市场四个数据叠加
    sum_matrix = list_sum_perdata(sum_matrix, MF_DFA(market2_3))
    sum_matrix = list_sum_perdata(sum_matrix, MF_DFA(market2_4))
    sum_matrix = np.array(sum_matrix) * 0.25
    hurst_matrix.append(sum_matrix)

    sum_matrix = MF_DFA(market2_5)  #每分钟成交额
    hurst_matrix.append(sum_matrix)
    sum_matrix = MF_DFA(market2_6)  #每分钟成交量
    hurst_matrix.append(sum_matrix)

    sum_matrix = MF_DFA(market1_1)  #复权单位净值
    hurst_matrix.append(sum_matrix)
    sum_matrix = MF_DFA(market1_2)  #贴水
    hurst_matrix.append(sum_matrix)
    sum_matrix = MF_DFA(market1_3)  #贴水率
    hurst_matrix.append(sum_matrix)
    sum_matrix = MF_DFA(market1_4)  #增长率
    hurst_matrix.append(sum_matrix)
    hurst_matrix = np.array(hurst_matrix)

    # 计算折溢价率和跟踪误差 --------------
    c1 = calculate_premium_discount_rate(numpy_1)
    print(f'平均折溢价率: {c1:.4f}%')
    c2 = calculate_tracking_error(numpy_1)
    print(f'跟踪误差: {c2:.4f}%')
    # 修正H(q)矩阵 --------------
    # 先对每个H的值减去0.5，并取绝对值
    hurst_matrix = np.abs(hurst_matrix - 0.5)
    X = transform_to_vector(c1, c2)  # 计算向量X
    Y = np.matmul(hurst_matrix, X.reshape(-1, 1))  # 计算向量Y
    Y_aggregated = aggregate_and_scale(Y, 4)  # 确保有4个聚合值

    Y_aggregates = np.mean(Y)
    Y_aggregated = np.exp(np.arctan(Y_aggregates) * (2 / np.pi) - 1) + Y_aggregated  # 确保有4个聚合值
    #e1, e2, e3, e4 = Y_aggregated  # 得到4个修正参数

    # 时间序列及处理，用作训练 --------------
    y = numpy_1[:, 0]  # 这个关系不要混乱了！我们最后要取得的是y
    x = numpy_1[:, 1]

# 开始交易
    new_profit, total_profit_chart, total_profit_rate, max_drawdown_result, predicted_y, origin_ys = gT0(params, x, y, Y_aggregated, profit, origin_eff_num)
    total_profit_chart = np.array(total_profit_chart)   #总利润
    total_profit_rate = np.array(total_profit_rate)     #平均利润率
    predicted_y = np.array(predicted_y)                 #预测值
    origin_ys = np.array(origin_ys)
    total_chart_bundled.append(total_profit_chart)
    total_rate_bundled.append(total_profit_rate)
    predicted_y_bundled.append(predicted_y)
    origin_y_bundled.append(origin_ys)

    process2_profit_dict[integer] = new_profit
    max_drawdown_dict[integer] = max_drawdown_result
    with open('process2_10_profitdict.json', 'w') as json_file:
        json.dump(process2_profit_dict, json_file, indent=4)  # 写入到文件中去
    with open('process2_10_maxdrawdown.json', 'w') as json_file:
        json.dump(max_drawdown_dict, json_file, indent=4)
    print("JSON_Dumped")

    np.save('process2_10_chart_bundled.npy', total_chart_bundled)
    np.save('process2_10_rate_bundled.npy', total_rate_bundled)
    np.save('process2_10_predict_y_bundled.npy',predicted_y_bundled)
    np.save('process2_10_origin_y_bundled.npy', origin_y_bundled)
    top_10_entries(process2_profit_dict, "process2_result_top10_comp.txt")
print("Done!")



#-------------------------------------------------------------------------------------

#-----------------------------------------
titles = ['513160.SH', '513390.SH', '512520.SH', '513290.SH', '159509.SZ',
          '159699.SZ', '159519.SZ', '512360.SH', '513970.SH', '159711.SZ']
x_labels = ['Trade Time/min'] * 10
y_labels = ['Unit Profit/(CNY/min$^{-1}$)'] * 10
# 显示图形
plt.show()
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
# 显示图形
plt.show()


#-----------------------------------------
titles = ['513160.SH', '513390.SH', '512520.SH', '513290.SH', '159509.SZ',
          '159699.SZ', '159519.SZ', '512360.SH', '513970.SH', '159711.SZ']
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

#-----------------------------------------


# 确定行数和列数
rows, cols = 2, 5
# 创建 2 行 5 列的子图
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 8))
#fig.suptitle('Comparison of Predicted and Original ETF Data')
titles = ['513160.SH', '513390.SH', '512520.SH', '513290.SH', '159509.SZ',
          '159699.SZ', '159519.SZ', '512360.SH', '513970.SH', '159711.SZ']
# 为了方便图例只显示一次，我们使用 handles 和 labels
lines = []
labels = []
# 遍历每一行数据绘制折线图
for i in range(rows * cols):
    row = i // cols
    col = i % cols
    ax = axes[row, col]
    # 绘制预测数据折线图
    line1, = ax.plot(predicted_y_bundled[i], 'r-', label='Predicted')
    # 绘制原始数据折线图
    line2, = ax.plot(origin_y_bundled[i], 'b-', label='Original')

    # 为了避免图例在每个子图中重复出现，我们只在第一次时添加它们
    if i == 0:
        lines.append(line1)
        lines.append(line2)
        labels.append(line1.get_label())
        labels.append(line2.get_label())

    # 设置子图标题等
    ax.set_title(titles[i])
    ax.set_xlabel('Trade Time/min')
    ax.set_ylabel('Market Price/CNY')

# 设置图例，只显示一次
fig.legend(lines, labels, loc='upper right')

# 调整子图的位置
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # 保证标题和子图之间有足够的间隔

# 显示图表
plt.show()
