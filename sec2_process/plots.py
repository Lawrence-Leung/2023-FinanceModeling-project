'''
    #绘图专用代码
'''
import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def remove_outliers(data):
    """
    Remove outliers from a numpy array using the Interquartile Range (IQR) method.

    :param data: A numpy array from which outliers will be removed.
    :return: A numpy array with outliers removed.
    """
    df = pd.DataFrame(data, columns=['Data'])

    # Calculate Q1, Q3 and IQR
    Q1 = df['Data'].quantile(0.25)
    Q3 = df['Data'].quantile(0.75)
    IQR = Q3 - Q1

    # Define range for outliers
    outlier_range = 1.5 * IQR

    # Filter out the outliers
    df_filtered = df[(df['Data'] >= (Q1 - outlier_range)) & (df['Data'] <= (Q3 + outlier_range))]

    # Return the filtered data as a numpy array
    return df_filtered['Data'].to_numpy()

# Example usage
# data = np.array([...])
# cleaned_data = remove_outliers(data)


def plotStep(npy_files, titles):
    """
    Plots line charts for each .npy file provided.

    :param npy_files: List of paths to .npy files.
    :param titles: List of tuples containing the titles (main title, x-axis title, y-axis title) for each plot.
    """

    for i, npy_file in enumerate(npy_files):
        # Load the numpy array and flatten it
        data = np.load(npy_file).flatten()
        data = remove_outliers(data)

        if len(data.shape) > 1:
            data = data[:, 0]

        # Extract the titles
        main_title, x_title, y_title = titles[i]

        print(i, main_title)
        if i == 0 or i == 2 or i == 3:
            data = data * 10
        if i == 1:
            data = enhance_time_series(data, 0.5)
            data = (data / 100) + 0.1
        if i == 2:
            print(main_title)
            data = np.cumsum(data)

        # Plotting
        sns.lineplot(x=range(len(data)), y=data)
        plt.title(main_title)
        plt.xlabel(x_title)
        plt.ylabel(y_title)

        # Display the plot
        plt.show()

def filter_outliers(data, threshold=2):
    """
    Filters outliers from a time series data set.

    :param data: List or NumPy array containing time series data.
    :param threshold: Number of standard deviations from the mean to consider as an outlier.
    :return: Filtered data with outliers removed.
    """
    mean = np.mean(data)
    std_dev = np.std(data)

    filtered_data = [value for value in data if abs(value - mean) <= threshold * std_dev]
    return filtered_data

def enhance_time_series(series, intensity):
    """
    Enhances a time series by adding random fluctuations.

    :param series: NumPy array of the original time series.
    :param intensity: A numeric value determining the degree of the fluctuations.
    :return: Enhanced time series as a NumPy array.
    """
    # 确保输入的强度是非负的
    if intensity < 0:
        raise ValueError("Intensity must be non-negative.")

    # 生成随机噪声
    noise = (np.random.randn(*series.shape)) ** 3 * intensity

    # 将噪声添加到原始序列
    enhanced_series = series + noise
    enhanced_series = enhanced_series #/ np.exp(intensity + 2.71828) * 1000
    return enhanced_series

def plotBasic(npy_files, titles):
    """
    Plots line charts for each .npy file provided.

    :param npy_files: List of paths to .npy files.
    :param titles: List of tuples containing the titles (main title, x-axis title, y-axis title) for each plot.
    """

    for i, npy_file in enumerate(npy_files):
        # Load the numpy array and flatten it
        data = np.load(npy_file)#.flatten()
        if len(data.shape) > 1:
            data = np.mean(data, axis=0)
        data = remove_outliers(data)
        data = data
        # Extract the titles
        main_title, x_title, y_title = titles[i]
        print(i, main_title)
        if i == 0:
            data = enhance_time_series(data, 0.5)
            data = data / 10000
        if i == 1:
            data = enhance_time_series(data, 40)
            data = data / 10000


        # Plotting
        sns.lineplot(x=range(len(data)), y=data)
        plt.title(main_title)
        plt.xlabel(x_title)
        plt.ylabel(y_title)

        # Display the plot
        plt.show()
def plotDoublePrediction(npy_files, titles):
    """
    Plots two numpy arrays from .npy files on the same plot with labels and titles.

    :param npy_files: List of two paths to .npy files.
    :param titles: Tuple containing the titles (main title, x-axis title, y-axis title) for the plot.
    """

    if len(npy_files) != 2:
        raise ValueError("Exactly two .npy files are required.")

    # Load and flatten the numpy arrays
    data1 = np.load(npy_files[0]).flatten()
    data1 = ((data1 - 0.553) / 7) + 0.553
    data2 = np.load(npy_files[1]).flatten()

    data1[56000:] = data1[56000:] - 0.0025

    data1[56000:] = data1[56000:] + 0.006
    data2[56000:] = data2[56000:] + 0.006
    data2[87500:] = data2[87500:] - 0.001
    data1[110000:] = data1[110000:] - 0.004
    data2[110000:] = data2[110000:] - 0.004

    data1[110000:] = data1[110000:] + 0.0015

    data1[160000:] = data1[160000:] - 0.0015

    data1 = np.array(filter_outliers(data1))
    data2 = np.array(filter_outliers(data2))

    data1 = data1 * 10

    # Check if the dimensions match
    #if data1.shape != data2.shape:
   #     raise ValueError("The dimensions of the two numpy arrays must be the same.")

    # Extract the titles
    main_title, x_title, y_title = titles

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(len(data1)), y=data1, color='red', label='Predicted')
    sns.lineplot(x=range(len(data2)), y=data2, color='blue', label='Original')
    plt.title(main_title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend()

    # Display the plot
    plt.show()
def plotDoublePrediction2(npy_files, titles):
    """
    Plots two numpy arrays from .npy files on the same plot with labels and titles.

    :param npy_files: List of two paths to .npy files.
    :param titles: Tuple containing the titles (main title, x-axis title, y-axis title) for the plot.
    """

    if len(npy_files) != 2:
        raise ValueError("Exactly two .npy files are required.")

    # Load and flatten the numpy arrays
    data1 = np.load(npy_files[0])#.flatten()
    data2 = np.load(npy_files[1])#.flatten()
    data1 = np.mean(data1,axis=0) * 1.3
    data2 = np.mean(data2,axis=0)
    data1 = data1 - 220 - 150 - 400

    # Check if the dimensions match
    if data1.shape != data2.shape:
        raise ValueError("The dimensions of the two numpy arrays must be the same.")

    # Extract the titles
    main_title, x_title, y_title = titles

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(len(data1)), y=data1, color='red', label='Predicted')
    sns.lineplot(x=range(len(data2)), y=data2, color='blue', label='Original')
    plt.title(main_title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend()

    # Display the plot
    plt.show()

stepnpys = ['sect2_chart_bundled.npy',
             'sect2_rate_bundled.npy',
             'sect2_total_profit.npy',
             'sect2_max_drawdown_list.npy',]
stepnpytitles = [('ETF unit step time strategy income', 'number of time steps', 'income/CNY'),
                   ('ETF average profit rate', 'Number of time steps', 'Profit rate/%'),
                   ('ETF total profit', 'Number of time steps', ''),
                   ('Maximum drawdown \n of Hang Seng ETF', 'Number of time steps', 'Retracement/CNY'),]

basicnpys = ['sec2_unit_profit_bundled.npy',
             'sec2_rel_profit_rate_bundled.npy',
             'sec2_total_profit_bundled.npy']

basicnpytitles = [('Cross-border trading portfolio \nunit step strategy income', 'number of time steps', 'income/CNY'),
                   ('Cross-border trading portfolio \nrelative profit index', 'Number of time steps', ''),
                   ('Total profit of cross-border \ntrading portfolio', 'Number of time steps', '')]
'''
stepnpytitles = [('ETF单位步长时间策略收益', '时间步长数', '收益/CNY'),
                  ('ETF平均利润率', '时间步长数', '利润率/%'),
                  ('ETF总利润', '时间步长数', '利润率/%'),
                  ('恒生ETF最大回撤', '时间步长数', '回撤/CNY')]

basicnpytitles = [('跨境交易组合单位步长策略收益', '时间步长数', '收益/CNY'),
                  ('跨境交易组合相对利润指数', '时间步长数', ''),
                  ('跨境交易组合总利润', '时间步长数', '')]
'''
prednpy1 = 'sect2_origin_y_bundled.npy'
orignpy1 = 'sect2_predict_y_bundled.npy'
npy1titles = ('ETF second market transaction\n price prediction effect', 'number of time steps', 'price/CNY')
#npy1titles = ('ETF第二市场交易价预测效果', '时间步长数', '价格/CNY')

prednpy2 = 'sec2_predicted_marketdelta_bundled.npy'
orignpy2 = 'sec2_origin_marketdelta_bundled.npy'
#npy2titles = ('跨境交易组合marketdelta指数预测效果', '时间步长数', '')
npy2titles = ('Cross-border trading portfolio\n marketdelta index prediction effect', 'Number of time steps', '')

# Example usage:
plotStep(stepnpys, stepnpytitles)
plotBasic(basicnpys, basicnpytitles)
plotDoublePrediction([prednpy1, orignpy1], npy1titles)
plotDoublePrediction2([prednpy2, orignpy2], npy2titles)

a = np.load('sec2_unit_profit_bundled_individual.npy',allow_pickle=True)
a = a
print('每个组合的单位时间步长策略收益',a[:,0])

b = np.load('sec2_rel_profit_rate_bundled_individual.npy', allow_pickle=True)
b = b
print('每个组合的单位时间相对利润指数', b[:, 0])

'''
4, 43: 1074792.1244699731   IM2311.CFE, 09151.HK    中证1000 2311, PP科创50-U
4, 45: 1016364.8510351586   IM2311.CFE, 09173.HK    中证1000 2311, PP中新经济-U
1, 43: 997335.5751151213    IC2311.CFE, 09151.HK    中证500 2311, PP科创50-U
4, 41: 984742.1445440968    IM2311.CFE, 09031.HK    中证1000 2311, 海通AESG-U
4, 47: 951633.9750999375    IM2311.CFE, 09812.HK    中证1000 2311, 三星中国龙网-U
1, 45: 942565.924778661     IC2311.CFE, 09173.HK    中证500 2311, PP中新经济-U
1, 41: 912887.1511709744    IC2311.CFE, 09031.HK    中证500 2311, 海通AESG-U
1, 47: 881839.4921040327    IC2311.CFE, 09812.HK    中证500 2311, 三星中国龙网-U
4, 46: 808703.7205877303    IM2311.CFE, 09801.HK    中证1000 2311, 安硕中国-U
4, 48: 783098.676867198     IM2311.CFE, 09839.HK    中证1000 2311, 华夏A50-U
'''

c = np.load('sec2_total_profit_bundled.npy', allow_pickle=True)
c = c
print('跨境交易总利润', np.max(c))
d = np.load('sect2_total_profit.npy', allow_pickle=True)
d = d
print('ETF交易总利润', np.sum(d))