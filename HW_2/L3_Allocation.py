import pandas as pd                     # To load data, we use the package pandas
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# Load the data

df = pd.read_excel('../Data/Factors_July26_July11.xlsx')        # It has 5 columns:  date, mkt, size, b/m, riskree rate
                                                       # downloaded from Ken French's website
df2 = pd.read_excel('../Data/Indu10_July26_July11.xlsx')       # return on 5 industry portfolios from Ken French's web
# print(df.head(),df2.head())
mkt = df.loc[:, "mkt"] / 100  # Mkt excess return;  divided by 100 due to data are in %
rf = df.loc[:, "rate"] / 100
R1 = df2.loc[:, 'Indu1': ] / 100  # extract the 5 industry returns, R1 is T by 5.

R1 = np.array(R1)  # convert list to array to apply np.functions, T x 5
rf = np.array(rf)  # convert list to array to apply np.functions, a T-vector
# note: T, is not Tx1 in Python as the latter is 2-dim
T = len(df)  # The number of observations
Re = np.ones((T, 10))  # creat storage for excess returns
# no need, but more clear
for i in range(10):
    Re[:, i] = R1[:, i] - rf  # the excess return:  each indu substracts riskfree rate, Re[:,i]-rf

mu5 = np.mean(Re, axis=0)  # the mean taking each column of the matrix, a row vector of 5
mu5 = mu5.T  # make it a column vector
V5 = np.cov(Re.T)  # the covariance estimate, 5 by 5
VI = np.linalg.inv(V5)  # The inverse of V
# The optimal weights on the 5 risky aasets
gamma = 3  # The risk-averse coeff.
w5 = (1 / gamma) * VI @ mu5  # an alternative:  np.matmul(VI, mu5)
#           matmul does matrix multiplication
#   @ is new and seems simpler and easier to see.
mu = mkt.mean()  # The expected mkt excess return
sig2 = mkt.var()  # The var of the mkt excess return
sigma = np.sqrt(sig2)  # Its vol
w = (1 / gamma) * mu / sig2  # The optimal weight on mkt
print('Rsik avrersion and Optimal wight on the market \n')
print('        {0:.4f}  {1:.4f}  \n '.format(gamma, w))
print('The Optimal wights on the 5 industries \n')
print(w5)
w_rf = 1 - np.dot(w5, np.ones((10, 1)))
print('The rest is on riskfree asset \n')
print(w_rf)