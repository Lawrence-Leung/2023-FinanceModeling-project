import os
import pandas as pd
import numpy as np
import json
#功能代码1-------------------------------------------------
'''
def process_and_save(directory_path):
    files = [f for f in os.listdir(directory_path) if
             os.path.isfile(os.path.join(directory_path, f)) and f.endswith('.xlsx')]

    for file in files:
        file_path = os.path.join(directory_path, file)
        # 读取Excel文件中的数据
        data = pd.read_excel(file_path, usecols='B:G', skiprows=1)
        # 将数据转换为numpy数组
        data_array = data.to_numpy()
        # 获取文件名，不包括扩展名
        file_name = os.path.splitext(file)[0]
        # 保存numpy数组到文件
        np.save(os.path.join(directory_path, file_name + '.npy'), data_array)
# 指定要处理的文件夹路径
directory_path = 'original_dataset'
# 调用函数
process_and_save(directory_path)
'''

#功能代码2-------------------------------------------------
'''
#已知一个.json文件，这个文件内有大量键值对。这些键值对的键为各种债券的名称，如"159507.SZ"，而值为一个整数的string，如"8"。
#请你将每一个债券的.xlsx文件中的4列、若干行数据，提取到一个numpy矩阵中，
#这个numpy矩阵也是4列、若干行的。然后，将这个numpy矩阵根据对应列第一行的债券名称，
#按照.json文件的键值对，将这个numpy矩阵命名为对应键的值，如"8"，并输出到文件夹中。

# 读取.xlsx文件和.json文件
xlsx_file = "process2_market1_data.xlsx"  # 替换为你的.xlsx文件路径
json_file = "name_extracted.json"   # 替换为你的.json文件路径

# 读取Excel文件
df = pd.read_excel(xlsx_file, header=0)

# 读取JSON文件
with open(json_file, 'r') as file:
    bond_names = json.load(file)

# 处理每个ETF债券的数据
for i in range(1, len(df.columns), 4):  # 从第2列开始，每4列处理一次
    # 提取每个ETF债券的4列数据
    etf_data = df.iloc[3:, i:i+4]  # 从第4行开始提取数据
    etf_matrix = etf_data.to_numpy()  # 转换为numpy矩阵

    # 获取债券名称
    bond_name = df.columns[i].split()[0]  # 假设债券名称在第一行，列名的第一个词

    # 检查债券名称是否在JSON文件中
    if bond_name in bond_names:
        # 将numpy矩阵输出到文件
        output_filename = f'dataset2/{bond_names[bond_name]}.npy'  # 输出文件名
        print(output_filename)
        np.save(output_filename, etf_matrix)  # 保存numpy矩阵
    else:
        print(f"债券名称 {bond_name} 不在JSON文件中。")

print("处理完成。")
'''
#功能代码3--------------------------------------------------
'''
# 文件路径
xlsx_file = 'process2_2_allETF_tradecode_names.xlsx'  # 替换为你的.xlsx文件路径
txt_file = 'process2_100ETFnames.txt'      # 替换为你的.txt文件路径
output_file = 'process2_2_100ETF_tradecodes.txt'       # 输出文件的名称

def extract_common_etf_codes(excel_path, txt_path, output_path):
    # 读取Excel文件，假设第一列是交易代码，第二列是ETF名称，且没有列标题
    df = pd.read_excel(excel_path, header=None)
    codes_column = 0  # 假设第一列是交易代码
    names_column = 1  # 假设第二列是ETF名称

    # 读取txt文件
    with open(txt_path, 'r', encoding='utf-8') as file:
        txt_etf_names = [line.strip() for line in file.readlines()]

    # 查找共有的ETF名称
    common_names = set(df[names_column]).intersection(txt_etf_names)

    # 根据共有的名称提取交易代码
    common_codes = df[df[names_column].isin(common_names)][codes_column]

    # 将交易代码写入新的txt文件
    with open(output_path, 'w', encoding='utf-8') as file:
        for code in common_codes:
            file.write(str(code) + '\n')

# 使用示例
extract_common_etf_codes(xlsx_file, txt_file, output_file)
# 请在你的本地环境运行这个脚本，并根据你的文件路径进行调整。
'''

#功能代码4，里面的函数后续会用上--------------------------------------------------
import json

def extract_etf_values(txt_path, json_input_path, json_output_path):
    # 读取txt文件中的ETF基金名称
    with open(txt_path, 'r', encoding='utf-8') as file:
        etf_names = [line.strip() for line in file.readlines()]

    # 读取json文件
    with open(json_input_path, 'r', encoding='utf-8') as file:
        etf_data = json.load(file)

    # 提取匹配的ETF值
    matching_etf_values = {name: etf_data[name] for name in etf_names if name in etf_data}

    # 输出到新的json文件
    with open(json_output_path, 'w', encoding='utf-8') as file:
        json.dump(matching_etf_values, file, ensure_ascii=False, indent=4)

def read_etf_json_to_list(json_path):
    # 读取json文件并将值导出到列表中
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    values = data.values()
    newlist = []
    for value in values:
        newlist.append(int(value))
    return newlist

# 使用示例
#extract_etf_values('process2_2_100ETF_tradecodes.txt', 'name_extracted.json', 'process2_3_100ETFindexes.json')
#etf_values_list = read_etf_json_to_list('process2_3_100ETFindexes.json')
#print(etf_values_list)

#功能代码5，--------------------------------------------------------
import os

def process_folder_2_2(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder_path, filename)
            print(file_path)
            data = np.load(file_path)

            # 检查数组是否至少有两列
            if data.shape[1] >= 2:
                # 遍历除了最后一行之外的所有行
                for i in range(data.shape[0] - 1):
                    if data[i, 1] == 0:  # 如果第二列的值为0
                        data[i, 1] = data[i + 1, 1]  # 用同一列下一行的数据替换

            # 保存修改后的文件
            np.save(file_path, data)

# 慎用！
#process_folder_2_2('dataset1')

#功能代码6
def find_min_rows_in_npy_files(folder_path):
    min_rows = None

    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            file_path = os.path.join(folder_path, file)
            data = np.load(file_path)
            rows = data.shape[0]

            if min_rows is None or rows < min_rows:
                min_rows = rows

    return min_rows

'''
# 使用示例
folder_path = 'dataset1'  # 替换为你的文件夹路径
rows_3 = find_min_rows_in_npy_files(folder_path) # 需要替换成实际的rows_3行数
print(f"The minimum number of rows in the .npy files is: {rows_3}")

dataset1_path = 'dataset1'
dataset2_path = 'dataset2'

def expand_dataset1_files(dataset1_path):
    for filename in os.listdir(dataset1_path):
        file_path_1 = os.path.join(dataset1_path, filename)
        print(file_path_1)
        if filename.endswith(".npy"):
            data_1 = np.load(file_path_1)
            # 假设 data_1 原本有 2 列，现在扩展到 6 列
            expanded_data = np.zeros((data_1.shape[0], 6))  # 6列，rows_3行
            expanded_data[:, :2] = data_1  # 保留原有的 2 列数据
            np.save(file_path_1, expanded_data)

def copy_data_from_dataset2(dataset1_path, dataset2_path):
    for filename in os.listdir(dataset2_path):
        file_path_2 = os.path.join(dataset2_path, filename)
        file_path_1 = os.path.join(dataset1_path, filename)
        if filename.endswith(".npy") and os.path.exists(file_path_1):
            data_2 = np.load(file_path_2, allow_pickle=True)
            data_1 = np.load(file_path_1, allow_pickle=True)

            for i in range(min(data_1.shape[0], data_2.shape[0])):
                start_row = 1 + i * 241 - 1  # 从 0 开始计数
                end_row = 241 * (i + 1)
                data_1[start_row:end_row, 2:6] = np.tile(data_2[i, :], (241, 1))

            np.save(file_path_1, data_1)
'''
# 先扩展 dataset1 中的文件
#expand_dataset1_files(dataset1_path)
# 从 dataset2 复制数据到 dataset1
#copy_data_from_dataset2(dataset1_path, dataset2_path,)\

#功能代码7
import matplotlib.pyplot as plt

# 加载数据
predict_y = np.load('process2_10_predict_y_bundled.npy')
origin_y = np.load('process2_10_origin_y_bundled.npy')

# 确定行数和列数
rows, cols = 2, 5

# 创建 2 行 5 列的子图
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 8))
fig.suptitle('Comparison of Predicted and Original ETF Data')

# 为了方便图例只显示一次，我们使用 handles 和 labels
lines = []
labels = []

# 遍历每一行数据绘制折线图
for i in range(rows * cols):
    row = i // cols
    col = i % cols
    ax = axes[row, col]

    # 绘制预测数据折线图
    line1, = ax.plot(predict_y[i], 'r-', label='Predicted')
    # 绘制原始数据折线图
    line2, = ax.plot(origin_y[i], 'b-', label='Original')

    # 为了避免图例在每个子图中重复出现，我们只在第一次时添加它们
    if i == 0:
        lines.append(line1)
        lines.append(line2)
        labels.append(line1.get_label())
        labels.append(line2.get_label())

    # 设置子图标题等
    ax.set_title(f'ETF {i+1}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')

# 设置图例，只显示一次
fig.legend(lines, labels, loc='upper right')

# 调整子图的位置
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # 保证标题和子图之间有足够的间隔

# 显示图表
plt.show()
