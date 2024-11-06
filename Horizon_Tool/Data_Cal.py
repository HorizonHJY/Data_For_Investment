import numpy as np

def calculate_sigma(data):
    """
    计算数据的标准差 sigma。
    参数:
        data (array-like): 输入的数据，可以是一个列表或 numpy 数组。
    返回:
        float: 数据的标准差 sigma。
    """
    variance = np.var(data)  # 计算方差
    sigma = np.sqrt(variance)  # 计算标准差
    return sigma

def calculate_metrics(returns, risk_free_rate):
    """
    计算平均回报率、标准差和夏普比率。
    参数:
        returns (array-like): 输入的回报率数据，可以是一个列表或 numpy 数组。
        risk_free_rate (float): 无风险利率（年化）。
    返回:
        tuple: (mean_return_annualized, std_dev_annualized, sharpe_ratio) 的元组，
               分别是年化的平均回报率、年化标准差和夏普比率。
    """
    # 计算月度平均回报率
    mean_return_monthly = np.mean(returns)
    # 年化平均回报率
    mean_return_annualized = (1 + mean_return_monthly) ** 12 - 1
    # 计算月度标准差
    std_dev_monthly = np.std(returns, ddof=1)
    # 年化标准差
    std_dev_annualized = std_dev_monthly * np.sqrt(12)
    # 计算夏普比率
    sharpe_ratio = (mean_return_annualized - risk_free_rate) / std_dev_annualized
    return mean_return_annualized, std_dev_annualized, sharpe_ratio