'''
    最终高频交易正式模型
    2023/11/11
'''
import numpy as np
import numpy.polynomial.polynomial as poly  #多重分形离散分析（MF-DFA）方法
from scipy.integrate import simps
from scipy.optimize import minimize_scalar
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import coint
import json

#任务2函数------------------------------------------------------
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
def gT0(params, x, y, Y_aggregated, profit = 0, origin_eff_num = 0):
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

    for i in range(len(y) - n_input_steps - n_output_steps + 1):
        #if i % 2000 == 0:
        #    print(i)
        # 将提取的数据添加到列表中
        extracted_time_series_data.append(y[i:i + n_input_steps])
    extracted_time_series_data = np.array(extracted_time_series_data)
    extracted_new_y_data = model.predict(extracted_time_series_data, verbose=0, batch_size=1000, workers=-1)  # 批量预测
    extracted_new_y_data -= 0.1450      #这个必须做修正，否则会出问题!
    #print(extracted_new_y_data)

    for t in range(len(extracted_time_series_data)):     #模拟所有周期的数据
        # 预测时间序列中t+1, t+2, t+3时刻的值
        # if t % 500 == 0:
        # print(extracted_time_series_data[t])
        # new_y_data = model.predict(extracted_time_series_data[t], batch_size=1000, verbose=0, workers=-1)
        # print(new_y_data)
        delta_y = corrected_price_prediction(extracted_new_y_data[t]) - y[t]  # 修正为一个值
        #print(delta_y)
        absolute_delta_y = y[t + 1] - y[t]  # delta_y

        profit_rate = (absolute_delta_y / y[t])  # 利润率
        max_y = max(max_y, y[t])
        min_y = min(min_y, y[t])

        # if origin_eff_num > 0: #必须得满足任何时候ETF持仓不为负
        if delta_y >= abs(trade_threshold):  # 折价交易
            trade_1st_market = e1 * d1 * math.log(abs(delta_y))  # 买入
            trade_2nd_market = e2 * d2 * math.log(abs(delta_y))  # 卖出
            if t % 240 == 239:  # 最后一刻补仓/平仓！
                new1 = -(trade_1st_market / (trade_1st_market + trade_2nd_market)) * origin_eff_num
                new2 = -(trade_2nd_market / (trade_1st_market + trade_2nd_market)) * origin_eff_num
                trade_1st_market = new1
                trade_2nd_market = new2
            if trade_2nd_market <= 0.7 * numpy_1[t, 5]:  # 限制条件（假设单次交易数量占每分钟总交易数量的比例），进行交易
                origin_eff_num = origin_eff_num + trade_1st_market - trade_2nd_market  # 持仓数量更新
                temp_money = -(trade_1st_market * round(x[t],2)) + (trade_2nd_market * round(y[j],2))  # 直接的买入卖出
                temp_money = temp_money - (eff1 * trade_1st_market) - (trade_2nd_market * round(y[j],2) * eff3 * 0.01) - (
                        (trade_1st_market * round(x[t],2) + trade_2nd_market * round(y[j],2)) * (eff2 + eff4) * 0.01)
                # 考虑一级市场结算金额、二级市场交易佣金、印花税和市场冲击成本
            else:
                temp_money = 0

        elif delta_y <= -1 * abs(trade_threshold):  # 溢价交易
            trade_1st_market = e3 * d3 * math.log(abs(delta_y))  # 卖出
            trade_2nd_market = e4 * d4 * math.log(abs(delta_y))  # 买入
            if t % 240 == 239:  # 最后一刻补仓/平仓！
                new1 = -(trade_1st_market / (trade_1st_market + trade_2nd_market)) * origin_eff_num
                new2 = -(trade_2nd_market / (trade_1st_market + trade_2nd_market)) * origin_eff_num
                trade_1st_market = new1
                trade_2nd_market = new2
            if trade_2nd_market <= 0.7 * numpy_1[t, 5]:  # 限制条件（假设单次交易数量占每分钟总交易数量的比例），进行交易
                origin_eff_num = origin_eff_num - trade_1st_market + trade_2nd_market  # 持仓数量更新
                temp_money = +(trade_1st_market * round(x[t],2)) - (trade_2nd_market * round(y[j],2))  # 直接的买入卖出
                temp_money = temp_money - (eff1 * trade_1st_market) - (trade_2nd_market * round(y[j],2) * eff3 * 0.01) - (
                        (trade_1st_market * round(x[t],2) + trade_2nd_market * round(y[j],2)) * (eff2 + eff4) * 0.01)
                # 考虑一级市场结算金额、二级市场交易佣金、印花税和市场冲击成本
            else:
                temp_money = 0

        else:  # 不进行交易
            temp_money = 0
            # else:
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
        #if j % 50 == 49:
            #print(j, "Profit=" ,profit)
    return profit, total_profit_chart, total_profit_rate, max_drawdown, predict_y, origin_y
#params: [d1, d2, d3, d4]
#x、y：原始的数据矩阵
#实际执行部分------------------------------------------------------
#交易参数
[d1, d2, d3, d4] = [15.993214229999998, -6.653766326666666, 5.151237160000001, 62.70280344]
params = [d1, d2, d3, d4]
trade_threshold = 5 * 1e-2  # 交易阈值信号限制：0.05元
profit2 = 0  # ETF交易
profit3 = 0  # 跨境交易
eff1 = 1  # ETF申购、赎回费用、过户费、证管费用、经手费、证券结算金:固定成本（元）（一级市场）
eff2 = 0.11  # 印花税率（百分比%）（所有市场） #已经加入股票费率
eff3 = 0.2  # ETF二级市场交易佣金、股票交易佣金百分比%）（二级市场）
eff4 = 0.05  # 市场冲击成本（百分比%）（所有市场）
trade_regulation = 0.7  # 假设单次交易数量占每分钟总交易数量的比例
origin_eff_num = 1000000  # 假设初始持仓数量100万ETF

#任务3函数------------------------------------------------------
# 数据预处理函数
def create_dataset(data, n_past, n_future):
    return np.array([data[i - n_past:i] for i in range(n_past, len(data) - n_future + 1)]), np.array([data[i:i + n_future] for i in range(n_past, len(data) - n_future + 1)])

def extract_rows_by_index(matrix, index_list):
    """
    Extract rows from a numpy matrix based on the provided index list.

    Parameters:
    matrix (numpy.ndarray): The original matrix.
    index_list (list): A list of row indices to extract.

    Returns:
    numpy.ndarray: A new matrix with the extracted rows.
    """
    # 验证索引是否在矩阵的行数范围内
    if max(index_list) >= matrix.shape[0]:
        raise ValueError("Index out of range.")

    # 使用numpy的高级索引提取行
    extracted_matrix = matrix[index_list, :]

    return extracted_matrix

# 构建LSTM模型的函数
def build_model(input_shape, output_units):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=input_shape))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=output_units))
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

# 训练模型的函数
def train_model(model, X_train, y_train, X_val, y_val):
    return model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=0)

# 预测未来数据的函数
def predict_future(data, model, n_past, n_future, scaler):
    last_sequence = data[-n_past:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(last_sequence.reshape(-1, 1))
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1)).flatten()
    next_sequence = model.predict(last_sequence_scaled.reshape(1, n_past, 1), verbose=0, workers=-1)
    return scaler.inverse_transform(next_sequence).flatten()

#交易策略影响因子计算
# 计算两个时间序列的价格相关性
def calculate_price_correlation(x, y):
    correlation, p_value = pearsonr(x, y)
    return correlation, p_value
# 检查两个时间序列的协整关系
def check_cointegration(x, y):
    score, p_value, _ = coint(x, y)
    return score, p_value
# 计算两个时间序列的回归斜率
def calculate_regression_slope(x, y):
    x = sm.add_constant(x)  # 添加常数项
    # 确保数据是NumPy数组并且是正确的数据类型
    x = np.asarray(X).astype(np.float64)
    y = np.asarray(Y).astype(np.float64)

    # 检查并处理NaN值
    if np.isnan(x).any() or np.isnan(y).any():
        # 处理NaN值，例如通过填充或删除
        x = np.nan_to_num(x)  # 将NaN替换为0，但这可能不是最佳的处理方式
        y = np.nan_to_num(y)

    # 检查并处理无穷值
    if np.isinf(x).any() or np.isinf(y).any():
        # 处理inf值
        x = np.where(np.isinf(x), 0, x)  # 将inf替换为0，但这可能不是最佳的处理方式
        y = np.where(np.isinf(y), 0, y)
    model = sm.OLS(y, x)
    results = model.fit()
    return results.params[0]  # 回归斜率
# 计算成交量的比率
def calculate_volume_ratio(x_quantity, y_quantity):
    return np.mean(y_quantity) / np.mean(x_quantity)
# 变换运算，6个因子输出4个因子，作为乘积因子
def calculate_decision_factors(c1, c2, c3, c4, c5, c6):
    c1 = abs(c1); c2 = abs(c2); c3 = abs(c3); c4 = abs(c4); c5 = abs(c5); c6 = abs(c6)
    # 使用矩阵整体的统计特性来计算决策因子
    d1 = (c1 + c2 + c3 + c4 + c5 + c6) / 6  # 矩阵所有元素的平均值
    return d1, d1, d1, d1

#优化算法
def objective_function(params):
    profit, _, _, _, _ = g(X, Y, market_delta_all, unitprice_diff_index, params)
    return -profit
# 真实交易策略
def g(X, Y, market_delta_all, unitprice_diff_index, params):
    profit = 0  #当前利润
    prev_profit = 0     #上一时刻利润
    prev_value_market_delta = 0    #上一时刻的“市场波动差异指数”
    hold_num_etf = 0    #当前时刻跨境etf持仓数量
    hold_num_ind = 0    #当前时刻股指期货持仓数量
    d1, d2, d3, d4 = params
    X_max = X[n_past - 1]
    X_min = X[n_past - 1]
    Y_max = Y[n_past - 1]
    Y_min = Y[n_past - 1]

    rel_profit_rate_bundled = []    #相对利润率list
    unit_profit_bundled = []    #单位时间收益list

    # 验证输入的时间序列X和Y的长度是否相等
    if len(X) != len(Y):
        raise ValueError("The lengths of X and Y must be equal.")
    # 验证market_delta_all的长度是否为X和Y的长度减8
    if len(market_delta_all) != len(X) - (n_past + 2):
        raise ValueError("The length of market_delta_all must be the length of X (or Y) minus 8.")
    # 定义结果列表
    results = []

    # 从t=6开始，直到n-3（因为我们需要包括n-3，所以循环到n-2）
    for t in range(n_past, len(X) - 2):
        # 从各个输入数组中获取相应的值
        X_price = X[t]  #国内市场单位标的资产下股指期货的价格
        Y_price = Y[t]  #国际市场单位标的资产下股指期货的价格
        value_market_delta = market_delta_all[t - n_past]
        X_price = round(X_price, 2)
        Y_price = round(Y_price, 2)

        if abs(value_market_delta) < abs(trade_threshold) or t == len(X) - 3:  # 进入了非套利区间或最后一刻，马上清仓
            profit = profit + (hold_num_ind * X_price *(1-0.01*(eff2_dom+eff3_dom+eff4_dom))\
                     + hold_num_etf * Y_price * (1-0.01*((eff2_abr+eff3_abr+eff4_abr))) \
                     - hold_num_ind * eff1_dom - hold_num_etf * eff1_abr) / 50
            # 假设可以马上清仓，不受交易数量限制
        elif value_market_delta >= abs(trade_threshold):    # 国际溢价
            if f1 * d1 > trade_regulation_dom:      #交易数量限制
                a = 0
            else:
                a = np.log(abs(f1 * d1) + 1) / 350  #正确修正哈哈哈！
            if f2 * d2 > trade_regulation_abr:
                b = 0
            else:
                b = np.log(abs(f2 * d2 / unitprice_diff_index) + 1) / 350

            profit = profit - (a * X_price) + (b * Y_price) #利润更新
            hold_num_etf = hold_num_etf - a #etf数量更新
            hold_num_ind = hold_num_ind + b #股指期货数量更新
            profit = profit - (eff1_dom * a) - (a * X_price) * 0.01 * (eff2_dom + eff3_dom + eff4_dom) \
                            - (eff1_abr * b) - (b * Y_price) * 0.01 * (eff2_abr + eff3_abr + eff4_abr)  #考虑到交易规则
        elif value_market_delta <= -abs(trade_threshold):   # 国际折价
            if f3 * d3 > trade_regulation_dom:      #交易数量限制
                a = 0
            else:
                a = np.log(abs(f3 * d3)) / 350
            if f4 * d4 > trade_regulation_abr:
                b = 0
            else:
                b = np.log(abs(f4 * d4 / unitprice_diff_index)) / 350

            profit = profit + (a * X_price) - (b * Y_price) #利润更新
            hold_num_etf = hold_num_etf + a #etf数量更新
            hold_num_ind = hold_num_ind - b #股指期货数量更新
            profit = profit - (eff1_dom * a) - (a * X_price) * 0.01 * (eff2_dom + eff3_dom + eff4_dom) \
                            - (eff1_abr * b) - (b * Y_price) * 0.01 * (eff2_abr + eff3_abr + eff4_abr)  #考虑到交易规则

        if X_max < X[t]:
            X_max = X[t]
        if X_min > X[t]:
            X_min = X[t]
        if Y_max < Y[t]:
            Y_max = Y[t]
        if Y_min > Y[t]:
            Y_min = Y[t]

        #相对收益率：这个收益率指的是以国内市场为基准、国际市场的变化所造成的收益率
        rel_profit_rate = Y[t] - X[t] / X[t]
        unit_profit_bundled.append(profit - prev_profit)    #添加单位时间收益
        rel_profit_rate_bundled.append(rel_profit_rate)

        prev_value_market_delta = value_market_delta
        prev_profit = profit
    # 退出循环后
    max_drawdown_x = float((X_max - X_min) / X_max)
    max_drawdown_y = float((Y_max - Y_min) / Y_max)
    return profit, unit_profit_bundled, rel_profit_rate_bundled, max_drawdown_x, max_drawdown_y

##-----------------------------------------------------------------
##-----------------------------------------------------------------
##-----------------------------------------------------------------
#大函数嵌套，每次开始运行这个东西
model = load_model('process2_100_lstm_model.h5')
model.summary()

#lists 2 ————————————————————————————————————————————————
process2_profit_list = []
max_drawdown_list = []
total_chart_bundled = []
total_rate_bundled = []
predicted_y_bundled = []
origin_y_bundled = []
rel_profit_rate_bundled = []  # 相对利润率list
unit_profit_bundled = []  # 单位时间收益list

#lists 3 ————————————————————————————————————————————————
combination_unit_profit_bundled = []
combination_rel_profit_rate_bundled = []
comb_predicted_marketdelta_bundled = []
comb_origin_marketdelta_bundled = []
combination_profitlist = []
combination_maxdraw_x_list = []
combination_maxdraw_y_list = []

profit3s = 0     #外部
max_drawdown_xs = 0.0   #外部
max_drawdown_x = 0.0    #内部
max_drawdown_ys = 0.0   #外部
max_drawdown_y = 0.0    #内部
unit_profit_bundled_b = []
rel_profit_rate_bundled_b = []
market_origin_all_b = []
market_delta_all_b = []

unit_profit_bundled_c = [[], [], [], [], [], [], [], [], [], []]
rel_profit_rate_bundled_c = [[], [], [], [], [], [], [], [], [], []]
X = []
Y = []
market_delta_all = []
unitprice_diff_index = []
#lists end ----------------------------------------------

np_step = 100   #步长
numpy_length_test = np.load('endprocess/dataset1/503.npy', allow_pickle=True)
np_length = numpy_length_test.shape[0]

for globali in range(0, np_length - np_length % np_step, np_step):
    ### 任务2模型-----------------------------------------------
    globali2 = int(globali / np_step)
    print(globali2 , ' out of ' , np_length / np_step)
    # 从文件中读取数据 --------------
    trade_threshold = 0  # 交易阈值信号限制：0.05元
    numpy_1_origin = insert_integer_into_string('endprocess/dataset1/.npy', 503, 20)    #按照索引读取numpy数据
    #print("Modified String:", numpy_1_origin)
    numpy_2_origin = insert_integer_into_string('endprocess/dataset2/.npy', 503, 20)
    #print("Modified String:", numpy_2_origin)
    numpy_original_origin = insert_integer_into_string('endprocess/original_dat/.npy', 503, 24)

    numpy_1 = np.load(numpy_1_origin, allow_pickle=True)
    numpy_1 = numpy_1[globali2 * np_step: min(globali2 * np_step + np_step, np_length), :]
    numpy_2 = np.load(numpy_2_origin, allow_pickle=True)
    numpy_2 = numpy_2[globali2 * np_step : min(globali2 * np_step + np_step - 1, np_length), :]
    numpy_original = np.load(numpy_original_origin, allow_pickle=True)
    numpy_original = numpy_original[globali2 * np_step : min(globali2 * np_step + np_step - 1, np_length), :]

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
    #print(f'平均折溢价率: {c1:.4f}%')
    c2 = calculate_tracking_error(numpy_1)
    #print(f'跟踪误差: {c2:.4f}%')
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
    #y = np.random.uniform(-0.2, 0.2, size=y.shape)
    x = numpy_1[:, 1]

    # 开始交易
    new_profit, total_profit_chart, total_profit_rate, max_drawdown_result, predicted_y, origin_ys = gT0(params, x, y, Y_aggregated, profit2, origin_eff_num)


    total_profit_chart = np.array(total_profit_chart)   #总利润
    total_profit_rate = np.array(total_profit_rate)     #平均利润率
    predicted_y = np.array(predicted_y)                 #预测值
    origin_ys = np.array(origin_ys)
    total_chart_bundled.append(total_profit_chart)
    total_rate_bundled.append(total_profit_rate)
    predicted_y_bundled.append(predicted_y)
    origin_y_bundled.append(origin_ys)

    process2_profit_list.append(new_profit)             #加入每个阶段的新数据
    max_drawdown_list.append(max_drawdown_result)       #加入每个阶段的新数据
    #print("JSON_Dumped - 2")
    np.save('sect2_chart_bundled.npy', total_chart_bundled)
    np.save('sect2_rate_bundled.npy', total_rate_bundled)
    np.save('sect2_predict_y_bundled.npy',predicted_y_bundled)
    np.save('sect2_origin_y_bundled.npy', origin_y_bundled)
    np.save('sect2_total_profit.npy', process2_profit_list)
    np.save('sect2_max_drawdown_list.npy', max_drawdown_list)

    ### 任务3模型-----------------------------------------------
    # 设定文件夹路径
    domestic_folder = 'endprocess/origin_dom'
    transnational_folder = 'endprocess/tr_etfs_dat'
    # 获取文件夹内所有的.npy文件
    domestic_files = [f for f in os.listdir(domestic_folder) if f.endswith('.npy')]
    transnational_files = [f for f in os.listdir(transnational_folder) if f.endswith('.npy')]
    # 按照文件名排序，确保顺序是从1开始的连续编号
    domestic_files.sort(key=lambda x: int(x.split('.')[0]))
    transnational_files.sort(key=lambda x: int(x.split('.')[0]))
    # 神经网络的输入、输出时间序列长度
    n_past = 7
    n_future = 3
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 交易策略决定因子
    [f1, f2, f3, f4] = [0.098, 0.120, 0.103, 0.100]   #确定好了
    params = [f1, f2, f3, f4]

    # 国内交易因子（股指期货）
    eff1_dom = 0  # ETF申购、赎回费用、过户费、证管费用、经手费、证券结算金:固定成本（元）(不适用）
    eff2_dom = 0.205  # 印花税率（百分比%）        百分比都是基于交易额度而言的。     #已经加入保证金、交易费率
    eff3_dom = 0.2  # ETF二级市场交易佣金、股票交易佣金百分比%）
    eff4_dom = 0.05  # 市场冲击成本（百分比%）
    trade_regulation_dom = 0.7  # 假设单次交易数量占每分钟总交易数量的比例（流动性限制指标）

    # 国际市场交易因子（跨境etf）
    eff1_abr = 0.71781  # 交易手续费:固定成本（元）（1美元在2023.10.31为7.1781元）
    eff2_abr = 0.11  # ETF的管理费用（百分比%）（所有市场） #已经加入股票
    eff3_abr = 0.1  # ETF二级市场交易佣金、股票交易佣金百分比%）（二级市场）
    eff4_abr = 0.02  # 价差（百分比%）（所有市场）
    trade_regulation_abr = 0.7  # 假设单次交易数量占每分钟总交易数量的比例（流动性限制指标）

    # 注意这个过程中需要注意比例！比例！
    # 遍历domestic_options_data文件夹中的.npy文件
    bundle_index = 0
    bundle_prop = [0.101, 0.089, 0.158, 0.099, 0.073, 0.079, 0.107, 0.093, 0.107, 0.094]    #比例操作

    for (a, b) in [(4, 43), (4, 45), (1, 43), (4, 41), (4, 47), (1, 45), (1, 41), (1, 47), (4, 46), (4, 48)]:        #外层循环，4个股指期货
        file_name = insert_integer_into_string('.npy', a, 0)
        trans_file_name = insert_integer_into_string('.npy', b, 0)

        i1 = int(file_name.split('.')[0])  # 获取编号i1
        domestic_file_path = os.path.join(domestic_folder, file_name)
        origin_domestic_npy = np.load(domestic_file_path, allow_pickle=True)
        origin_domestic_npy = origin_domestic_npy[globali2 * np_step : min(globali2 * np_step + np_step - 1, np_length), :]

        if origin_domestic_npy.shape[1] < 5:
            print("The array doesn't have enough columns to perform this operation.")
        else:
            # 计算每行前4列的平均值， axis=1 表示沿着每一行操作
            X = np.mean(origin_domestic_npy[:, :4], axis=1)
        X_quantity = origin_domestic_npy[:, 4]     #X股指期货的交易量

        # 在每个i1.npy文件下，再遍历transnational_etfsdata文件夹中的.npy文件
        #for trans_file_name in transnational_files:     #内层循环，100个跨境ETF
        i2 = int(trans_file_name.split('.')[0])  # 获取编号i2
        transnational_file_path = os.path.join(transnational_folder, trans_file_name)
        origin_transetf_npy = np.load(transnational_file_path, allow_pickle=True)
        origin_transetf_npy = origin_transetf_npy[globali2 * np_step : min(globali2 * np_step + np_step - 1, np_length), :]

        if bundle_index < 10:
            bundle_mul = bundle_prop[bundle_index] #乘积

        if origin_transetf_npy.shape[1] < 2:
            print("The array doesn't have enough columns to perform this operation.")
        else:
            # 计算每行前4列的平均值 axis=1 表示沿着每一行操作
            Y = np.mean(origin_transetf_npy[:, :4], axis=1)
            Y_quantity = origin_transetf_npy[:, 4]
            unitprice_diff_index = Y[0] / X[0]     # 每个组合的“价格差异指数”
            Y_origin = Y
            Y = Y / unitprice_diff_index           # 归一化
            marketdelta = X - Y                # 每个组合的“市场波动差异指数”

        #推理用脚本（已经load了model）
        data_scaled = scaler.fit_transform(marketdelta.reshape(-1, 1))
        market_x, market_origin_debug = create_dataset(data_scaled.flatten(), n_past, n_future)
        market_delta_result = model.predict(market_x, workers=-1, verbose=0)
        market_delta_all = []
        market_origin_all = []
        for i3 in range(market_delta_result.shape[0]):
            market_delta_all.append(corrected_price_prediction(market_delta_result[i3]))
            market_origin_all.append(market_origin_debug[i3, 0])
        market_delta_all = np.array(market_delta_all)
        market_origin_all = np.array(market_origin_all)

        #计算影响因子，区分好X、Y！
        correlation, corr_p_value = calculate_price_correlation(X, Y)
        cointegration_score, coint_p_value = check_cointegration(X, Y)
        slope = calculate_regression_slope(X, Y)
        volume_ratio = calculate_volume_ratio(X_quantity, Y_quantity)
        #unitprice_diff_index（不是用于参数判断）

        c1 = correlation            #皮尔森相关系数
        c2 = corr_p_value           #相关性统计检验p值
        c3 = cointegration_score    #协整测试得分
        c4 = coint_p_value          #协整测试p值
        c5 = slope                  #回归斜率
        c6 = volume_ratio           #成交量比率
        d1, d2, d3, d4 = calculate_decision_factors(c1, c2, c3, c4, c5, c6)

        d1 = d1 * bundle_mul       #乘以比例
        d2 = d2 * bundle_mul
        d3 = d3 * bundle_mul
        d4 = d4 * bundle_mul

        profit3, unit_profit_bundled, rel_profit_rate_bundled, max_drawdown_x, max_drawdown_y \
            = g(X, Y, market_delta_all, unitprice_diff_index, params)   #真正的运算函数

        #bounds = [(0.08, 0.12), (0.08, 0.12), (0.08, 0.12), (0.08, 0.12)]
        #result = minimize(objective_function, params, bounds=bounds, method='BFGS')
        #params = result.x
        #print(result.x)
        #print(i1, ", ", i2)
        #print("A trading simulation has done.")
        #print(profit3)

        #每次运行完保存文件--------------------------------
        unit_profit_bundled = np.array(unit_profit_bundled)         #打包好的每组合单位时间收益，画图用
        rel_profit_rate_bundled = np.array(rel_profit_rate_bundled) #打包好的每组合相对利润率，画图用
        profit3s = profit3s + profit3
        with open('process3_1_combination_profitdict.json', 'w') as json_file:
            json.dump(combination_profitlist, json_file, indent=4)
        max_drawdown_xs = max(max_drawdown_xs, max_drawdown_x)
        max_drawdown_ys = max(max_drawdown_ys, max_drawdown_y)
        if len(unit_profit_bundled_b) == 0:
            unit_profit_bundled_b = unit_profit_bundled
        else:
            unit_profit_bundled_b += unit_profit_bundled
        if len(rel_profit_rate_bundled_b) == 0:
            rel_profit_rate_bundled_b = rel_profit_rate_bundled
        else:
            rel_profit_rate_bundled_b += rel_profit_rate_bundled
        if len(market_origin_all_b) == 0:
            market_origin_all_b = market_origin_all * bundle_mul
        else:
            market_origin_all_b += market_origin_all * bundle_mul
        if len(market_delta_all_b) == 0:
            market_delta_all_b = market_delta_all * bundle_mul
        else:
            market_delta_all_b += market_delta_all * bundle_mul
        unit_profit_bundled_c[bundle_index]= unit_profit_bundled
        rel_profit_rate_bundled_c[bundle_index] = rel_profit_rate_bundled

        bundle_index = bundle_index + 1

    combination_profitlist.append(profit3s)   #每组合的总利润，按照周期操作
    combination_maxdraw_x_list.append(max_drawdown_xs)  # 每组合的国内股指期货最大回撤率
    combination_maxdraw_y_list.append(max_drawdown_ys)  # 每组合的国际跨境ETF最大回撤率
    combination_unit_profit_bundled.append(unit_profit_bundled_b) #每组合单位时间收益
    combination_rel_profit_rate_bundled.append(rel_profit_rate_bundled_b) #每组合相对利润率
    comb_origin_marketdelta_bundled.append(market_origin_all_b)    #原始的market_delta
    comb_predicted_marketdelta_bundled.append(market_delta_all_b)  #预测的market_delta

    np.save('sec2_unit_profit_bundled.npy', combination_unit_profit_bundled)    #总单位时间收益，时间序列
    np.save('sec2_rel_profit_rate_bundled.npy', combination_rel_profit_rate_bundled)    #总相对利润率，时间序列
    np.save('sec2_origin_marketdelta_bundled.npy', comb_origin_marketdelta_bundled)     #平均原始market_delta
    np.save('sec2_predicted_marketdelta_bundled.npy', comb_predicted_marketdelta_bundled)   #平均预测market_delta
    np.save('sec2_unit_profit_bundled_individual.npy', np.array(unit_profit_bundled_c))   #单个组合单位时间收益
    np.save('sec2_rel_profit_rate_bundled_individual.npy', rel_profit_rate_bundled_c)   #单个组合相对利润率
    np.save('sec2_maxdraw_x_bundled.npy', combination_maxdraw_x_list)   #总组合股指期货最大回撤率
    np.save('sec2_maxdraw_y_bundled.npy', combination_maxdraw_y_list)   #总组合国际跨境ETF最大回撤率
    np.save('sec2_total_profit_bundled.npy', combination_profitlist)    #总组合总利润，时间序列

    #print("Numpy Dumped")

