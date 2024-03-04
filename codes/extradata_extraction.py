import os
import numpy as np

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

#---------------------------------------
# 功能代码1
'''
# 设置输入和输出文件夹的路径
input_folder = 'original_dataset'
output_folder = 'dataset1'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有.npy文件
for file_name in os.listdir(input_folder):
    if file_name.endswith('.npy'):
        input_npy_name = os.path.join(input_folder, file_name)
        output_npy_name = os.path.join(output_folder, file_name)
        print(input_npy_name)

        # 调用你的函数
        process_and_save(input_npy_name, output_npy_name)
        process_and_update(input_npy_name, output_npy_name)

        # 如果需要，你可以在此处添加其他处理
'''
#---------------------------------------
# 功能代码2
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

'''
# 示例使用
integer = 123

# 执行函数
result_string = insert_integer_into_string('original_dataset/.npy', integer, 17)
print("Modified String:", result_string)
result_string = insert_integer_into_string('dataset1/.npy', integer, 9)
print("Modified String:", result_string)
'''

#---------------------------------------
# 功能代码3
'''
def calculate_centroid(coordinates):
    """
    计算四维坐标系中一组坐标的重心。

    :param coordinates: 一个包含多个四维坐标的列表，每个坐标都是形式为[a, b, c, d]的列表。
    :return: 重心坐标，格式为[a, b, c, d]。
    """
    if not coordinates:
        return [0, 0, 0, 0]

    # 初始化总和变量为四维坐标的每个维度
    sum_a, sum_b, sum_c, sum_d = 0, 0, 0, 0

    # 累加每个坐标的每个维度
    for a, b, c, d in coordinates:
        sum_a += a
        sum_b += b
        sum_c += c
        sum_d += d

    # 计算每个维度的平均值
    num_points = len(coordinates)
    centroid = [sum_a / num_points, sum_b / num_points, sum_c / num_points, sum_d / num_points]

    return centroid

# 示例使用
coordinates = [
    [4.55965422, -2.58208904, 1.00, 1.00],
    [28.85120055, -12.28013236 , 36.85171398,  74.48976689],
    [26.19189689, -11.33120911, 3.48920961, 82.1007765],
    [1.73203846 ,-0.46442561 ,-37.7800326  , 77.57448106],
    [8.24730462 ,-2.09102969, 17.02760942 , 59.48394212],
    [26.37719064 ,-11.17371215 ,10.31892255 ,81.56785407]]

centroid = calculate_centroid(coordinates)
print("The centroid of the coordinates is:", centroid)
'''

#功能代码4
'''
import matplotlib.pyplot as plt

# 假设你的数据是一个名为 data 的列表，包含10个一维数组
# 例如: data = [np.array([1, 2, 3, 4, 5]), np.array([5, 4, 3, 2, 1]), ...]
data = [np.random.rand(10) for _ in range(10)]  # 这只是一个示例，你应该用你自己的数据替换它
print(data)

# 设置图表的标题、x轴和y轴的标签
titles = ['Title 1', 'Title 2', 'Title 3', 'Title 4', 'Title 5', 'Title 6', 'Title 7', 'Title 8', 'Title 9', 'Title 10']
x_labels = ['Index'] * 10
y_labels = ['Value'] * 10

# 创建图形
fig, axs = plt.subplots(2, 5, figsize=(15, 6))

# 遍历每个子图并绘制数据
for i, ax in enumerate(axs.flat):
    print(i, ax)
    ax.plot(data[i])
    ax.set_title(titles[i])
    ax.set_xlabel(x_labels[i])
    ax.set_ylabel(y_labels[i])

# 调整子图间的空间以保证标签不重叠
plt.tight_layout()

# 显示图形
plt.show()
'''

# 功能代码5
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Concatenate, Flatten

# 假设我们有一些一维时间序列数据
time_series_data = np.random.rand(7)  # 14个时间点
hurst_matrix = np.random.rand(7, 7)  # 7x7的H(q)矩阵

# LSTM网络的参数
time_steps = 5  # 时间序列的时间步长
lstm_units = 50  # LSTM单元的数量

# 重塑时间序列数据以适应LSTM的输入格式 (samples, time_steps, features)
# 这里我们假设每个样本由连续的3个时间点组成
reshaped_time_series_data = np.array([time_series_data[i:i+time_steps] for i in range(len(time_series_data) - time_steps + 1)])
reshaped_time_series_data = reshaped_time_series_data[..., np.newaxis]  # 增加特征维度

# 时间序列输入
time_series_input = Input(shape=(time_steps, 1), name='time_series_input')
# LSTM层
lstm_out = LSTM(lstm_units)(time_series_input)
# Hurst矩阵输入
hurst_input = Input(shape=(7, 7), name='hurst_input')
hurst_flattened = Flatten()(hurst_input)  # 将Hurst矩阵展平
# 将LSTM输出和Hurst矩阵展平后的输出合并
merged = Concatenate()([lstm_out, hurst_flattened])
# 全连接层
dense_out = Dense(1, activation='linear')(merged)  # 假设我们的输出是一个连续值
# 构建模型
model = Model(inputs=[time_series_input, hurst_input], outputs=dense_out)
# 编译模型
model.compile(optimizer='adam', loss='mse')
# 打印模型结构
model.summary()

# 假设我们有一些目标值用于训练
target_values = np.random.rand(reshaped_time_series_data.shape[0], 1)
print("Target Values's Shape")
print(target_values.shape)
# 将Hurst矩阵扩展为与时间序列数据相同的批次大小
hurst_matrix_expanded = np.expand_dims(hurst_matrix, axis=0)
hurst_matrix_expanded = np.repeat(hurst_matrix_expanded, reshaped_time_series_data.shape[0], axis=0)

# 训练模型
model.fit([reshaped_time_series_data, hurst_matrix_expanded], target_values, epochs=10)
model.save('process2_3_lstm_model.h5')

print("Matrix Show")
print("Reshaped Time Series Data Shape", reshaped_time_series_data.shape)
print("Hurst Matrix Expanded Shape", hurst_matrix_expanded.shape)
print("Target Values Shape", target_values.shape)

print("Model Forwarding")
# 模型推理
# 假设我们有一些新的一维时间序列数据用于推理
new_time_series_data = np.random.rand(7)  # 5个新的时间点

# 重塑新的时间序列数据以适应LSTM的输入格式
reshaped_new_time_series_data = np.array([new_time_series_data[i:i+time_steps] for i in range(len(new_time_series_data) - time_steps + 1)])
reshaped_new_time_series_data = reshaped_new_time_series_data[..., np.newaxis]  # 增加特征维度

# 假设Hurst矩阵保持不变
# 如果Hurst矩阵有变化，需要提供新的Hurst矩阵

# 使用模型进行推理
model = load_model('process2_3_lstm_model.h5')
predicted_values = model.predict([reshaped_new_time_series_data, hurst_matrix_expanded])
print("Predicted Values Shape", predicted_values.shape)

# 输出的predicted_values就是模型对新时间序列数据的预测值
