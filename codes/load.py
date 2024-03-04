import numpy as np

def display_numpy_data(file_path):
    # 读取numpy数据
    data = np.load(file_path)

    # 显示数据到屏幕上
    print(data)

# 指定要读取的numpy文件路径
file_path = 'dataset1/1.npy'

# 调用函数
display_numpy_data(file_path)
