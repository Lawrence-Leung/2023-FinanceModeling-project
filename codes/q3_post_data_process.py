'''
    第3题 后处理代码
    2023/11/5
'''
import numpy as np

# 函数集合------------------------------------------------------
# 数据预处理函数
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

# 运行代码
rows_to_extract = [303, 305, 42, 301, 307, 44, 40, 46, 306, 308]
combination_unit_profit_bundled = np.load('process3_1_unit_profit_bundled.npy')
combination_rel_profit_rate_bundled = np.load('process3_1_rel_profit_rate_bundled.npy')
comb_origin_marketdelta_bundled = np.load('process3_1_origin_marketdelta_bundled.npy')
comb_predicted_marketdelta_bundled = np.load('process3_1_predicted_marketdelta_bundled.npy')

combination_unit_profit_bundled = extract_rows_by_index(combination_unit_profit_bundled, rows_to_extract)
combination_rel_profit_rate_bundled = extract_rows_by_index(combination_rel_profit_rate_bundled, rows_to_extract)
comb_origin_marketdelta_bundled = extract_rows_by_index(comb_origin_marketdelta_bundled, rows_to_extract)
comb_predicted_marketdelta_bundled = extract_rows_by_index(comb_predicted_marketdelta_bundled, rows_to_extract)
# 每个组合的收益率指标
np.save('process3_2_rel_profit_rate_bundled.npy', combination_rel_profit_rate_bundled)
comb_total_profit = [1074792.1244699731, 1016364.8510351586, 997335.5751151213, 984742.1445440968, 951633.9750999375,
                     942565.924778661, 912887.1511709744, 881839.4921040327, 808703.7205877303, 783098.676867198]
# 每个组合的总收益
# 回撤：x、y
comb_max_redraw_x = [0.06846177296958808, 0.06846177296958808, 0.07019219814024413, 0.06846177296958808, 0.06846177296958808,
                     0.07019219814024413, 0.07019219814024413, 0.07019219814024413, 0.06846177296958808, 0.06846177296958808]
comb_max_redraw_y = [0.06892137314665468, 0.08269596587206159, 0.0689213731466548, 0.07296979471821187, 0.09676027806856712,
                     0.08269596587206163, 0.07296979471821192, 0.09676027806856717, 0.08032063481589224, 0.07767127555386255]
for i in range(10):
    print(combination_rel_profit_rate_bundled[i, :])

# 计算指标：投资组合比例、组合风险、组合收益
import numpy as np

# 计算夏普比率
def calculate_sharpe_ratio(annual_return, annual_risk, risk_free_rate=0):
    return (annual_return - risk_free_rate) / annual_risk

# Markowitz模型的简化实现，这里不涉及求解二次优化问题
def markowitz_weights(returns, risk_aversion=1):
    cov_matrix = np.cov(returns)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    ones = np.ones(len(returns))
    weights = risk_aversion * inv_cov_matrix.dot(ones)
    return weights / weights.sum()

# 加权算法来决定最终的投资组合权重
def calculate_portfolio_weights(profit, max_redraw_x, max_redraw_y):
    sharpe_ratios = calculate_sharpe_ratio(np.mean(profit, axis=1), np.std(profit, axis=1))
    markowitz_weights_ = markowitz_weights(profit)
    combined_weights = (sharpe_ratios + markowitz_weights_) / 2
    normalized_weights = combined_weights / combined_weights.sum()
    return normalized_weights.tolist()

# 计算组合风险
def calculate_portfolio_risks(max_redraw_x, max_redraw_y):
    # 这里简单地使用最大回撤来估计风险，现实中可能会更复杂
    combined_risk = np.sqrt(np.array(max_redraw_x)**2 + np.array(max_redraw_y)**2)
    return combined_risk.tolist()

# 计算组合收益
def calculate_portfolio_returns(comb_total_profit):
    return comb_total_profit

# 主函数，计算投资组合权重，风险和收益
def calculate_portfolio_allocation(combination_rel_profit_rate_bundled, comb_total_profit, comb_max_redraw_x, comb_max_redraw_y):
    weights = calculate_portfolio_weights(combination_rel_profit_rate_bundled, comb_max_redraw_x, comb_max_redraw_y)
    risks = calculate_portfolio_risks(comb_max_redraw_x, comb_max_redraw_y)
    returns = calculate_portfolio_returns(comb_total_profit)
    return weights, risks, returns

# 示例调用
weights, risks, returns = calculate_portfolio_allocation(
    combination_rel_profit_rate_bundled,
    comb_total_profit,
    comb_max_redraw_x,
    comb_max_redraw_y
)

print("Weights:", weights)
print("Risks:", risks)
print("Returns:", returns)
