import pandas as pd
import numpy as np

# 读取数据
df = pd.read_excel('../Data/Factors_July26_May05.xlsx')  # 包含5列：date, mkt, size, b/m, riskfree rate
df2 = pd.read_excel('../Data/Indu10_July26_May05.xlsx')  # 包含10个行业投资组合的收益率数据

# 数据处理
mkt = df.loc[:, "mkt"] / 100          # 市场超额收益率，数据以%为单位，需要除以100
rf = df.loc[:, "rate"] / 100          # 无风险利率
R1 = df2.loc[:, 'Indu1':] / 100       # 提取10个行业的收益率，按列名范围选择，并转换为百分比
R1 = np.array(R1)                     # 转换为NumPy数组以便使用NumPy函数
rf = np.array(rf)                     # 转换为NumPy数组

# 初始化超额收益存储矩阵
T = len(df)                           # 观测数量
Re = np.ones((T, 10))                 # 创建存储超额收益的矩阵 (T x 10)

# 计算每个行业的超额收益
for i in range(10):
    Re[:, i] = R1[:, i] - rf          # 每个行业收益减去无风险利率

# 协方差矩阵估计
V10 = np.cov(Re.T)                    # 计算协方差矩阵（10 x 10）

# 输出协方差矩阵（放大10^4倍，便于比较）
print('Covariance Matrix * 10000:')
print(np.round(V10 * 10000, 2))       # 使用 np.round 限制输出的小数点位数

# 特征值和特征向量计算
eigvals, eigvecs = np.linalg.eig(V10)  # 使用NumPy计算特征值和特征向量

# 按特征值大小排序
idx = eigvals.argsort()[::-1]          # 获取降序排列的索引
eigvals_sorted = eigvals[idx]
eigvecs_sorted = eigvecs[:, idx]

# 输出特征值和特征向量（放大10^4倍）
print('\nEigenvalues * 10000 (scaled):')
print(np.round(eigvals * 10000, 2))

print('\nCorresponding Eigenvectors (unsorted):')
print(np.round(eigvecs, 4))            # 限制输出的小数点位数，增强可读性

print('\nSorted Eigenvalues * 10000 (scaled):')
print(np.round(eigvals_sorted * 10000, 2))

print('\nSorted Eigenvectors:')
print(np.round(eigvecs_sorted, 4))     # 输出排序后的特征向量


# 提取第一个和第二个主成分对应的特征向量
A1 = eigvecs_sorted[:, 0]  # 第一个主成分（第一个特征向量）的系数
A2 = eigvecs_sorted[:, 1]  # 第二个主成分（第二个特征向量）的系数，作为示例

# 计算每列的均值（行业收益的均值）
mu10 = np.mean(Re, axis=0)  # 每列的均值，1 x 10
mu10 = mu10.reshape(1, 10)  # 确保 mu10 为 1 x 10 的形状

# 创建一个 T x 1 的全1矩阵，用于去均值
onesT = np.ones((T, 1))  # T x 1 矩阵
RR = Re - onesT @ mu10   # 去均值的收益矩阵，T x 10

# 第一个主成分（PCA因子）的实现值
f = RR @ A1  # T x 1 矩阵，计算主成分因子值（类似于市场因子的实现值）

# 计算第一个特征值占总特征值的比例
ones10 = np.ones((1, 10))  # 1 x 10 的全1向量
Fraction = eigvals_sorted[0] / np.dot(ones10, eigvals_sorted)  # 第一个特征值占比

# 输出结果
print('The values of the first PCA factor in the first 3 periods:')
print('Similar to the market factor in the first 3 periods:\n')
print(f[0:3])  # 输出前3期的主成分因子值

print('\nThe fraction of the first eigenvalue relative to the total:')
print(Fraction)  # 输出第一个特征值占比
