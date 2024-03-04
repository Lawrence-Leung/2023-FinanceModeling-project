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

hurst_matrix = []
hurst_matrix.append(sum_matrix)

sum_matrix = MF_DFA(market2_5)  # 每分钟成交额
hurst_matrix.append(sum_matrix)
sum_matrix = MF_DFA(market2_6)  # 每分钟成交量
hurst_matrix.append(sum_matrix)

sum_matrix = MF_DFA(market1_1)  # 复权单位净值
hurst_matrix.append(sum_matrix)
sum_matrix = MF_DFA(market1_2)  # 贴水
hurst_matrix.append(sum_matrix)
sum_matrix = MF_DFA(market1_3)  # 贴水率
hurst_matrix.append(sum_matrix)
sum_matrix = MF_DFA(market1_4)  # 增长率
hurst_matrix.append(sum_matrix)
hurst_matrix = np.array(hurst_matrix)

# 先对每个H的值减去0.5，并取绝对值
hurst_matrix = np.abs(hurst_matrix - 0.5)
Y_aggregated = aggregate_and_scale(Y, 4)  # 确保有4个聚合值

c1 = calculate_premium_discount_rate(numpy_1)
print(f'平均折溢价率: {c1:.4f}%')
c2 = calculate_tracking_error(numpy_1)
print(f'跟踪误差: {c2:.4f}%')

X = transform_to_vector(c1, c2)  # 计算向量X
Y = np.matmul(hurst_matrix, X.reshape(-1, 1))  # 计算向量Y
Y_aggregated = aggregate_and_scale(Y, 4)  # 确保有4个聚合值

