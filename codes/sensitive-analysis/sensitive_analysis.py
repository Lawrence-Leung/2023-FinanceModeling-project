'''
    模型灵敏度分析运行脚本
    2023/11/7
'''
import numpy as np
import numpy.polynomial.polynomial as poly
from scipy.stats import levy_stable

def list_sum_perdata(list1, list2):
    #把两个数组按元素增加
    result = [x + y for x,y in zip(list1, list2)]
    return result
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
#折溢价率
def calculate_premium_discount_rate(numpy_1, numpy_2):
    # 提取二级市场价格和一级市场价格
    market_prices = numpy_1
    nav_prices = numpy_2

    # 替换零值以避免除以零的错误
    epsilon = 10  # 一个正数，由经验确定
    nav_prices_replaced = np.where(nav_prices == 0, epsilon, nav_prices)

    # 计算每分钟的折溢价率
    premium_discount_rates = ((market_prices - nav_prices_replaced) / nav_prices_replaced) * 100

    # 计算折溢价率的平均值
    average_premium_discount_rate = np.mean(premium_discount_rates)

    return average_premium_discount_rate
#跟踪误差
def calculate_tracking_error(numpy_1, numpy_2):
    # 提取二级市场价格和一级市场价格
    market_prices = numpy_1
    nav_prices = numpy_2

    # 确保没有零值，以避免除以零的错误
    epsilon = 2.71828  # 一个正数，由经验确定
    nav_prices_replaced = np.where(nav_prices == 0, epsilon, nav_prices)

    # 计算每分钟的价格差异比率
    price_diff_ratios = (market_prices - nav_prices_replaced) / nav_prices_replaced

    # 计算跟踪误差
    tracking_error = np.std(price_diff_ratios)
    return tracking_error
def generate_fractal_time_series(a, b, size=100):
    """
    Generates a 1x100 time series numpy array with fractal degree 'a' and volatility 'b'.
    The fractal degree is simulated using a generalized Hurst exponent approach,
    and the volatility is simulated using a Levy stable distribution.

    Parameters:
    a : float
        The fractal degree parameter (generalized Hurst exponent).
    b : float
        The volatility parameter.
    size : int, optional
        The size of the time series to generate, default is 100.

    Returns:
    np.array
        A numpy array of the generated time series.
    """

    # Generate a Levy stable distribution with alpha as the fractal degree
    # and beta as the skewness parameter. Scale and location are set to b and 0, respectively.
    time_series = levy_stable.rvs(alpha=a, beta=0, loc=0, scale=b, size=size)

    # Cumulative sum to get the time series from the increments
    #time_series = np.cumsum(time_series)

    # Reshape to a 1xN numpy array
    return time_series.reshape(1, -1)


def generate_time_series(mean, variance, length=100):
    """
    Generate a time series numpy array with given mean and variance.

    Parameters:
    mean (float): Mean of the time series.
    variance (float): Variance of the time series.
    length (int): Length of the time series array.

    Returns:
    numpy.ndarray: Generated time series.
    """
    # Standard deviation is the square root of the variance
    std_dev = np.sqrt(variance)

    # Generate a time series with the specified mean and standard deviation
    time_series = np.random.normal(loc=mean, scale=std_dev, size=length)

    return time_series

# Example usage:
#a = 0.5  # Fractal degree parameter
#b = 1    # Volatility parameter
inputdatas = []
eshow = np.zeros([1, 10, 10])
#for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#    for b in [0.5, 0.6, 0.7, 0.8, 0,9, 1.0, 1.1, 1.2, 1,3, 1.4]:
for a in range(10):
    for b in range(10):
        print(a, b)
        x_a = 10 + np.exp(float(a / 10) + 1)
        x_b = 10 + np.exp(float(b / 10) + 1)
        for i in range(9):
            time_series = generate_time_series(x_a, x_b)
            inputdatas.append(time_series)

        # 计算MF-DFA --------------
        hurst_matrix = []
        sum_matrix = []
        for k in range(7):
            sum_matrix = MF_DFA(inputdatas[k])
            hurst_matrix.append(sum_matrix)
        hurst_matrix = np.array(hurst_matrix) + x_a

        # 计算折溢价率和跟踪误差 --------------
        c1 = calculate_premium_discount_rate(inputdatas[7], inputdatas[8])
        print(f'平均折溢价率: {c1:.4f}%')
        c2 = calculate_tracking_error(inputdatas[7], inputdatas[8]) + x_b
        print(f'跟踪误差: {c2:.4f}%')

        hurst_matrix = np.abs(hurst_matrix - 0.5)
        X = transform_to_vector(c1, c2)  # 计算向量X
        Y = np.matmul(hurst_matrix, X.reshape(-1, 1))  # 计算向量Y
        Y_aggregated = aggregate_and_scale(Y, 4)

        Y_aggregates = np.mean(Y)
        Y_aggregated = np.exp(np.arctan(Y_aggregates) * (2 / np.pi) - 1) + Y_aggregated #确保有4个聚合值
        #e1, e2, e3, e4 = Y_aggregated  # 得到4个修正参数
        for j in range(1):
            eshow[j, a, b] = Y_aggregated[0]


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# 假设我们有等差数列形成的规律的网格数据作为x和y轴的坐标
x = np.linspace(0, 9, 10)
y = np.linspace(0, 9, 10)
x, y = np.meshgrid(x, y)
# 假设z是根据x和y通过某个函数计算得出的，这里我们用一个简单的二次函数作为例子
# 绘制3D曲面图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 使用plot_surface方法绘制曲面，我们也可以添加颜色映射（cmap）
surf = ax.plot_surface(x, y, eshow[0], cmap='viridis')

# 添加颜色条
#fig.colorbar(surf, shrink=0.5, aspect=5)

# 设置图表标题和坐标轴标签
ax.set_title('Sensitivity Analysis Result')
ax.set_xlabel('Fractality')
ax.set_ylabel('Variability')
ax.set_zlabel('')

# 显示图表
plt.show()

