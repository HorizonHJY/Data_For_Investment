import pandas as pd
import numpy as np
import statsmodels.api as sm

# 加载数据
# 股票数据（IBM）
ibm_df = pd.read_excel('../Data/IBM.xlsx')

# 市场数据
market_df = pd.read_excel('../Data/Returns_handbook_Python_data.xlsx', sheet_name='Monthly')

# 转换日期格式
# IBM 数据日期为 YYYYMMDD 格式，需要截取年份和月份
ibm_df['YearMonth'] = ibm_df['Date'].astype(str).str[:6].astype(int)

# 市场数据日期为 YYYYMM 格式
market_df['YearMonth'] = market_df['Date (yyyymm)']

# 合并两份数据，按 YearMonth 对齐
merged_df = pd.merge(ibm_df, market_df, on='YearMonth', how='inner')

# 提取回报率和市场回报率
stock_return = merged_df['Return']  # IBM 的回报率
market_return = merged_df['S&P 500 index']  # S&P 500 的回报率

# 市场无风险收益率
risk_free_rate = merged_df['3-month Treasury bill yield (secondary market)']

# 计算超额回报率
stock_excess_return = stock_return - risk_free_rate
market_excess_return = market_return - risk_free_rate

# 回归分析
y = stock_excess_return.values.reshape(-1, 1)  # 被解释变量
x = market_excess_return.values.reshape(-1, 1)  # 解释变量
x_with_const = sm.add_constant(x)  # 添加常数项

# 运行回归
reg = sm.OLS(y, x_with_const)
results = reg.fit()

# 显示结果
print("Regression Results:")
print(results.summary())

# 提取 Beta 和显著性
beta = results.params[1]  # Beta 系数
p_value = results.pvalues[1]  # Beta 的 p 值

if p_value < 0.05:
    print(f"Beta is significant at the 5% level. Beta: {beta:.4f}, p-value: {p_value:.4f}")
else:
    print(f"Beta is NOT significant at the 5% level. Beta: {beta:.4f}, p-value: {p_value:.4f}")
