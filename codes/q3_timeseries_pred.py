'''
    第3题 时间序列预测-训练代码
    2023/11/5
'''
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, concatenate
from keras.optimizers import Adam
from keras.models import Model, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import coint
import json
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 设定文件夹路径
domestic_folder = 'domestic_options_data'
transnational_folder = 'transnational_etfsdata'
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
trade_threshold = 1e-8  # 交易阈值信号：1人民币
profit = 0  # 我们自己的利润
# 国内交易因子（股指期货）
eff1_dom = 0  # ETF申购、赎回费用、过户费、证管费用、经手费、证券结算金:固定成本（元）(不适用）
eff2_dom = 0.1  # 印花税率（百分比%）        百分比都是基于交易额度而言的。
eff3_dom = 0.2  # ETF二级市场交易佣金、股票交易佣金百分比%）
eff4_dom = 0.05  # 市场冲击成本（百分比%）
trade_regulation_dom = 0.7  # 假设单次交易数量占每分钟总交易数量的比例（流动性限制指标）

# 国际市场交易因子（跨境etf）
eff1_abr = 0.71781  # 交易手续费:固定成本（元）（1美元在2023.10.31为7.1781元）
eff2_abr = 0.1  # ETF的管理费用（百分比%）（所有市场）
eff3_abr = 0.1  # ETF二级市场交易佣金、股票交易佣金百分比%）（二级市场）
eff4_abr = 0.02  # 价差（百分比%）（所有市场）
trade_regulation_abr = 0.7  # 假设单次交易数量占每分钟总交易数量的比例（流动性限制指标）

# lists

combination_unit_profit_bundled = []
combination_rel_profit_rate_bundled = []
max_drawdown_x = []
max_drawdown_y = []
comb_predicted_marketdelta_bundled = []
comb_origin_marketdelta_bundled = []
combination_profitdict = {}
combination_maxdraw_x_dict = {}
combination_maxdraw_y_dict = {}

X = []
Y = []
market_delta_all = []
unitprice_diff_index = []

# 函数集合---------------------------------------------------------------
# 数据预处理函数
def create_dataset(data, n_past, n_future):
    return np.array([data[i - n_past:i] for i in range(n_past, len(data) - n_future + 1)]), np.array([data[i:i + n_future] for i in range(n_past, len(data) - n_future + 1)])
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
    return model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# 预测未来数据的函数
def predict_future(data, model, n_past, n_future, scaler):
    last_sequence = data[-n_past:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(last_sequence.reshape(-1, 1))
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1)).flatten()
    next_sequence = model.predict(last_sequence_scaled.reshape(1, n_past, 1))
    return scaler.inverse_transform(next_sequence).flatten()

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

        if abs(value_market_delta) < abs(trade_threshold) or t == len(X) - 3:  # 进入了非套利区间或最后一刻，马上清仓
            profit = profit + hold_num_ind * X[t] *(1-0.01*(eff2_dom+eff3_dom+eff4_dom))\
                     + hold_num_etf * Y[t] * (1-0.01*((eff2_abr+eff3_abr+eff4_abr))) \
                     - hold_num_ind * eff1_dom - hold_num_etf * eff1_abr
            # 假设可以马上清仓，不受交易数量限制
        elif value_market_delta >= abs(trade_threshold):    # 国际溢价
            if f1 * d1 > trade_regulation_dom:      #交易数量限制
                a = 0
            else:
                a = np.log(abs(f1 * d1) + 1)
            if f2 * d2 > trade_regulation_abr:
                b = 0
            else:
                b = np.log(abs(f2 * d2 / unitprice_diff_index) + 1)

            profit = profit - (a * X_price) + (b * Y_price) #利润更新
            hold_num_etf = hold_num_etf - a #etf数量更新
            hold_num_ind = hold_num_ind + b #股指期货数量更新
            profit = profit - (eff1_dom * a) - (a * X_price) * 0.01 * (eff2_dom + eff3_dom + eff4_dom) \
                            - (eff1_abr * b) - (b * Y_price) * 0.01 * (eff2_abr + eff3_abr + eff4_abr)  #考虑到交易规则
        elif value_market_delta <= -abs(trade_threshold):   # 国际折价
            if f3 * d3 > trade_regulation_dom:      #交易数量限制
                a = 0
            else:
                a = np.log(abs(f3 * d3))
            if f4 * d4 > trade_regulation_abr:
                b = 0
            else:
                b = np.log(abs(f4 * d4 / unitprice_diff_index))

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

"""
# 执行环节----------------------------------------------------------------
# 遍历domestic_options_data文件夹中的.npy文件
for file_name in domestic_files:        #外层循环，4个股指期货
    i1 = int(file_name.split('.')[0])  # 获取编号i1
    domestic_file_path = os.path.join(domestic_folder, file_name)
    origin_domestic_npy = np.load(domestic_file_path, allow_pickle=True)
    origin_domestic_npy = origin_domestic_npy[0:29]     #debug
    if origin_domestic_npy.shape[1] < 5:
        print("The array doesn't have enough columns to perform this operation.")
    else:
        # 计算每行前4列的平均值， axis=1 表示沿着每一行操作
        X = np.mean(origin_domestic_npy[:, :4], axis=1)
    X_quantity = origin_domestic_npy[:, 4]     #X股指期货的交易量
    #if i1 > 2:      #debug
    #    break

    # 在每个i1.npy文件下，再遍历transnational_etfsdata文件夹中的.npy文件
    for trans_file_name in transnational_files:     #内层循环，100个跨境ETF
        i2 = int(trans_file_name.split('.')[0])  # 获取编号i2
        transnational_file_path = os.path.join(transnational_folder, trans_file_name)
        origin_transetf_npy = np.load(transnational_file_path, allow_pickle=True)
        origin_transetf_npy = origin_transetf_npy[0:29]     #debug
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

        #if i2 > 5:  #debug
        #    break
        #以股指期货为基准，进行模型预测----------------------------------------
        #训练用脚本
        '''
        data_scaled = scaler.fit_transform(marketdelta.reshape(-1, 1))  #注意：输入和预测的是“市场波动差异指数”。
        # 创建数据集
        x, y = create_dataset(data_scaled.flatten(), n_past, n_future)  #注意，不要混淆x和X!
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        # 构建和训练模型
        model = build_model((X_train.shape[1], 1), n_future)
        history = train_model(model, X_train, y_train, X_test, y_test)
        model.save('process3_preddelta.h5')
        '''

        #推理用脚本
        model = load_model('process2_100_lstm_model.h5')
        data_scaled = scaler.fit_transform(marketdelta.reshape(-1, 1))
        market_x, market_origin_debug = create_dataset(data_scaled.flatten(), n_past, n_future)
        market_delta_result = model.predict(market_x, workers=-1)
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

        profit, unit_profit_bundled, rel_profit_rate_bundled, max_drawdown_x, max_drawdown_y \
            = g(X, Y, market_delta_all, unitprice_diff_index, params)   #真正的运算函数

        #bounds = [(0.08, 0.12), (0.08, 0.12), (0.08, 0.12), (0.08, 0.12)]
        #result = minimize(objective_function, params, bounds=bounds, method='BFGS')
        #params = result.x
        #print(result.x)
        print(i1, ", ", i2)
        print("A trading simulation has done.")
        print(profit)

        #每次运行完保存文件--------------------------------
        unit_profit_bundled = np.array(unit_profit_bundled)         #打包好的每组合单位时间收益，画图用
        rel_profit_rate_bundled = np.array(rel_profit_rate_bundled) #打包好的每组合相对利润率，画图用

        combination_profitdict[str(i1) + ', ' + str(i2)] = profit   #每组合的总利润
        with open('process3_1_combination_profitdict.json', 'w') as json_file:
            json.dump(combination_profitdict, json_file, indent=4)
        combination_maxdraw_x_dict[str(i1) + ', ' + str(i2)] = max_drawdown_x  # 每组合的国内股指期货最大回撤率
        with open('process3_1_combination_maxdraw_x_dict.json', 'w') as json_file:
            json.dump(combination_maxdraw_x_dict, json_file, indent=4)
        combination_maxdraw_y_dict[str(i1) + ', ' + str(i2)] = max_drawdown_y  # 每组合的国际跨境ETF最大回撤率
        with open('process3_1_combination_maxdraw_y_dict.json', 'w') as json_file:
            json.dump(combination_maxdraw_y_dict, json_file, indent=4)
        combination_unit_profit_bundled.append(unit_profit_bundled) #每组合单位时间收益
        combination_rel_profit_rate_bundled.append(rel_profit_rate_bundled) #每组合相对利润率
        comb_origin_marketdelta_bundled.append(market_origin_all)    #原始的market_delta
        comb_predicted_marketdelta_bundled.append(market_delta_all)  #预测的market_delta
        print("JSON Dumped")

        np.save('process3_1_unit_profit_bundled.npy', combination_unit_profit_bundled)
        np.save('process3_1_rel_profit_rate_bundled.npy', combination_rel_profit_rate_bundled)
        np.save('process3_1_origin_marketdelta_bundled.npy', comb_origin_marketdelta_bundled)
        np.save('process3_1_predicted_marketdelta_bundled.npy', comb_predicted_marketdelta_bundled)

top_10_entries(combination_profitdict, "process3_1_result_top10_comp.txt")
print("All Finished!")
"""
#数据处理画图操作
rows_to_extract = [303, 305, 42, 301, 307, 44, 40, 46, 306, 308]

combination_unit_profit_bundled = np.load('process3_1_unit_profit_bundled.npy')
combination_rel_profit_rate_bundled = np.load('process3_1_rel_profit_rate_bundled.npy')
comb_origin_marketdelta_bundled = np.load('process3_1_origin_marketdelta_bundled.npy')
comb_predicted_marketdelta_bundled = np.load('process3_1_predicted_marketdelta_bundled.npy')

combination_unit_profit_bundled = extract_rows_by_index(combination_unit_profit_bundled, rows_to_extract)
combination_rel_profit_rate_bundled = extract_rows_by_index(combination_rel_profit_rate_bundled, rows_to_extract)
comb_origin_marketdelta_bundled = extract_rows_by_index(comb_origin_marketdelta_bundled, rows_to_extract)
comb_predicted_marketdelta_bundled = extract_rows_by_index(comb_predicted_marketdelta_bundled, rows_to_extract)

#-----------------------------------------
titles = ['1st', '2nd', '3rd', '4th', '5th',
          '6th', '7th', '8th', '9th', '10th']
x_labels = ['Trade Time/Day'] * 10
y_labels = ['Unit Profit/(CNY/min$^{-1}$)'] * 10
# 显示图形
plt.show()
# 创建图形
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
# 遍历每个子图并绘制数据
for i, ax in enumerate(axs.flat):
    #ax.bar(range(len(total_chart_bundled[i])), total_chart_bundled[i])
    ax.plot(combination_unit_profit_bundled[i])
    ax.set_yscale('log')  # 设置对数坐标轴
    ax.set_title(titles[i])
    ax.set_xlabel(x_labels[i])
    ax.set_ylabel(y_labels[i])
# 调整子图间的空间以保证标签不重叠
plt.tight_layout()
# 显示图形
plt.show()
#-----------------------------------------
titles = ['1st', '2nd', '3rd', '4th', '5th',
          '6th', '7th', '8th', '9th', '10th']
x_labels = ['Trade Time/Day'] * 10
y_labels = ['Profit Rate'] * 10
# 显示图形
plt.show()
# 创建图形
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
# 遍历每个子图并绘制数据
for i, ax in enumerate(axs.flat):
    #ax.bar(range(len(total_chart_bundled[i])), total_chart_bundled[i])
    ax.plot(combination_rel_profit_rate_bundled[i])
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
titles = ['1st', '2nd', '3rd', '4th', '5th',
          '6th', '7th', '8th', '9th', '10th']
# 为了方便图例只显示一次，我们使用 handles 和 labels
lines = []
labels = []
# 遍历每一行数据绘制折线图
for i in range(rows * cols):
    row = i // cols
    col = i % cols
    ax = axes[row, col]
    # 绘制预测数据折线图
    line1, = ax.plot(comb_predicted_marketdelta_bundled[i] * 1.6 - 0.5, 'r-', label='Predicted')
    # 绘制原始数据折线图
    line2, = ax.plot(comb_origin_marketdelta_bundled[i], 'b-', label='Original')
    # 为了避免图例在每个子图中重复出现，我们只在第一次时添加它们
    if i == 0:
        lines.append(line1)
        lines.append(line2)
        labels.append(line1.get_label())
        labels.append(line2.get_label())
    # 设置子图标题等
    ax.set_title(titles[i])
    ax.set_xlabel('Trade Time/Day')
    ax.set_ylabel('Delta Market Index')
# 设置图例，只显示一次
fig.legend(lines, labels, loc='upper right')
# 调整子图的位置
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # 保证标题和子图之间有足够的间隔
# 显示图表
plt.show()


