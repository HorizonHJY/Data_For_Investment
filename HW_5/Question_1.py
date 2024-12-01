import pandas as pd
import numpy as np
import statsmodels.api as sm

import pandas as pd
import numpy as np
import statsmodels.api as sm


def regression_analysis(y, x):
    """
    回归分析函数
    参数：
        y: 因变量（被解释变量）
        x: 自变量（解释变量）
    返回：
        回归结果、Beta 值、p 值
    """
    x_with_const = sm.add_constant(x)  # 添加常数项
    reg = sm.OLS(y, x_with_const)
    results = reg.fit()

    # 提取回归系数和显著性
    beta = results.params[1]
    p_value = results.pvalues[1]

    return results, beta, p_value


def question_1(merged_df):
    """
    问题 1: 使用市场回报 Rmkt,t 回归当前股票回报 Rt
    """
    stock_excess_return = merged_df['Return'] - merged_df['3-month Treasury bill yield (secondary market)']
    market_excess_return = merged_df['S&P 500 index'] - merged_df['3-month Treasury bill yield (secondary market)']
    y = stock_excess_return.values.reshape(-1, 1)
    x = market_excess_return.values.reshape(-1, 1)

    results, beta, p_value = regression_analysis(y, x)

    print("\nQuestion 1: Regression Results (Rt on Rmkt,t):")
    print(results.summary())
    if p_value < 0.05:
        print(f"Beta is significant at the 5% level. Beta: {beta:.4f}, p-value: {p_value:.4f}")
    else:
        print(f"Beta is NOT significant at the 5% level. Beta: {beta:.4f}, p-value: {p_value:.4f}")


def question_2(merged_df):
    """
    问题 2: 使用滞后的市场回报 Rmkt,t-1 回归当前股票回报 Rt
    """
    merged_df['Market_Excess_Return_Lag'] = merged_df['S&P 500 index'].shift(1) - merged_df[
        '3-month Treasury bill yield (secondary market)']
    merged_df = merged_df.dropna()  # 去除因滞后导致的缺失值

    stock_excess_return = merged_df['Return'] - merged_df['3-month Treasury bill yield (secondary market)']
    market_excess_return_lag = merged_df['Market_Excess_Return_Lag']
    y = stock_excess_return.values.reshape(-1, 1)
    x = market_excess_return_lag.values.reshape(-1, 1)

    results, beta, p_value = regression_analysis(y, x)

    print("\nQuestion 2: Regression Results (Rt on Rmkt,t-1):")
    print(results.summary())
    if p_value < 0.05:
        print(f"Beta is significant at the 5% level. Beta: {beta:.4f}, p-value: {p_value:.4f}")
    else:
        print(f"Beta is NOT significant at the 5% level. Beta: {beta:.4f}, p-value: {p_value:.4f}")


def question_3(merged_df):
    """
    问题 3: 使用滞后的股票回报 Rt-1 回归当前股票回报 Rt
    """
    merged_df['Stock_Excess_Return_Lag'] = merged_df['Return'].shift(1) - merged_df[
        '3-month Treasury bill yield (secondary market)']
    merged_df = merged_df.dropna()  # 去除因滞后导致的缺失值

    stock_excess_return = merged_df['Return'] - merged_df['3-month Treasury bill yield (secondary market)']
    stock_excess_return_lag = merged_df['Stock_Excess_Return_Lag']
    y = stock_excess_return.values.reshape(-1, 1)
    x = stock_excess_return_lag.values.reshape(-1, 1)

    results, beta, p_value = regression_analysis(y, x)

    print("\nQuestion 3: Regression Results (Rt on Rt-1):")
    print(results.summary())
    if p_value < 0.05:
        print(f"Beta is significant at the 5% level. Beta: {beta:.4f}, p-value: {p_value:.4f}")
    else:
        print(f"Beta is NOT significant at the 5% level. Beta: {beta:.4f}, p-value: {p_value:.4f}")


# 加载数据
ibm_df = pd.read_excel('../Data/IBM.xlsx')
market_df = pd.read_excel('../Data/Returns_handbook_Python_data.xlsx', sheet_name='Monthly')

# 合并数据
ibm_df['YearMonth'] = ibm_df['Date'].astype(str).str[:6].astype(int)
market_df['YearMonth'] = market_df['Date (yyyymm)']
merged_df = pd.merge(ibm_df, market_df, on='YearMonth', how='inner')

# 调用函数回答问题 1, 2, 3
# question_1(merged_df)
# question_2(merged_df)
question_3(merged_df)
