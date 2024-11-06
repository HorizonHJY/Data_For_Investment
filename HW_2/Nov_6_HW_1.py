import pandas as pd                     # To load data, we use the package pandas
import numpy as np
from dateutil.rrule import DAILY
import cvxopt
from Horizon_Tool.Data_Cal import *
import matplotlib.pyplot as plt
# %matplotlib inline
# Load the data
df = pd.read_excel('../Data/Factors_July26_July11.xlsx')        # It has 5 columns:  date, mkt, size, b/m, riskree rate
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
for i in range(10):
    Re[:, i] = R1[:, i] - rf  # the excess return:  each indu substracts riskfree rate, Re[:,i]-rf
mu10 = np.mean(Re, axis=0)  # the mean taking each column of the matrix, a row vector of 5
mu10 = mu10.T  # make it a column vector
# print(mu10)
V5 = np.cov(Re.T)  # the covariance estimate, 5 by 5
VI = np.linalg.inv(V5)  # The inverse of V
# The optimal weights on the 5 risky aasets
gamma = 3  # The risk-averse coeff.
w10 = (1 / gamma) * VI @ mu10  # an alternative:  np.matmul(VI, mu10)
#           matmul does matrix multiplication
#   @ is new and seems simpler and easier to see.
mu = mkt.mean()  # The expected mkt excess return
sig2 = mkt.var()  # The var of the mkt excess return
sigma = np.sqrt(sig2)  # Its vol
w = (1 / gamma) * mu / sig2  # The optimal weight on mkt
def P_f_f():
    print('Risk Aversion and Optimal Weight on the Market')
    print(f"Risk Aversion (gamma): {gamma:.4f}")
    print(f"Optimal Weight on the Market: {w:.4f}\n")
    print("The Optimal Weights on the 10 Industries")
    for i, weight in enumerate(w10):
        print(f"Industry {i + 1}: {weight:.4f}")
    print()
    w_rf = 1 - np.dot(w10, np.ones((10, 1)))
    print("The Rest is on Risk-Free Asset")
    print(f"Weight on Risk-Free Asset: {w_rf[0]:.4f}\n")
# P_f_f()
#P1_(2)
def P_f_s():
    Port = np.ones((T,))  # define this T-vector to store the returns on the portfolio
    # to be compatible with rf
    Port[0] = np.dot(w10, Re[0]) + rf[0]  # return in the first period, the weight on rf is absorbed
    print(Port.shape, w10.shape, rf.shape)
    # into the previous excess return term, see formulas in the slides
    for t in range(T):
        Port[t] = np.dot(w10, Re[t]) + rf[t]
    ExPort = Port - rf  # excess return of the optimla portfolio
    muP = ExPort.mean()
    sigP = calculate_sigma(ExPort)
    SharpeP = np.sqrt(12) * muP / sigP
    # print(ExPort.shape, Port.shape, rf.shape)  # double check the vectors are cpmpatible
    annualized_mean = (1 + muP) ** 12 - 1  # annualized mean
    annualized_std = sigP * np.sqrt(12)  # annualized std
    print("\nOptimal Portfolio Metrics:")
    print("Annualized Mean Return: {:.5f}".format(annualized_mean))
    print("Annualized Standard Deviation: {:.5f}".format(annualized_std))
    print("Annualized Sharpe Ratio: {:.5f}".format(SharpeP))
    print("-" * 50)
P_f_s()
def P_f_t():
    df = pd.read_excel('../Data/Factors_July26_July11.xlsx')
    mkt = df.loc[:, "mkt"] / 100
    mu = mkt.mean()  # The expected mkt excess return
    sig2 = mkt.var()  # The var of the mkt excess return
    sigma = np.sqrt(sig2)  # Its vol
    Sharpe = np.sqrt(12) * mu / sigma
    print("Market Portfolio Sharpe Ratio: {:.5f}".format(Sharpe))
    # Annualize the mean and standard deviation
    annualized_mean_mkt = (1 + mu) ** 12 - 1  # Annualized mean return
    annualized_std_mkt = sigma * np.sqrt(12)  # Annualized standard deviation
    Sharpe_mkt = np.sqrt(12) * mu / sigma  # Annualized Sharpe Ratio
    print("\nMarket Portfolio Metrics:")
    print("Annualized Mean Return: {:.5f}".format(annualized_mean_mkt))
    print("Annualized Standard Deviation: {:.5f}".format(annualized_std_mkt))
    print("Annualized Sharpe Ratio: {:.5f}".format(Sharpe_mkt))
    print("-" * 50)

def P_f_4():
    # Compute the accu returns of the opt port and the mkt
    Port = np.ones((T,))  # define this T-vector to store the returns on the portfolio
    # to be compatible with rf
    Port[0] = np.dot(w10, Re[0]) + rf[0]  # return in the first period, the weight on rf is absorbed
    # into the previous excess return term, see formulas in the slides
    for t in range(T):
        Port[t] = np.dot(w10, Re[t]) + rf[t]
    CC_Port = np.ones((T,))  # to store the accumulative returns
    CC = np.ones((T,))  # to store the accumulative returns
    mkt2 = mkt + rf  # Add back riskfree rate to get pure mkt return
    CC_Port[0] = 1 + Port[0]  # initial accu return
    CC[0] = 1 + mkt2[0]  # initial accu return
    for t in range(T - 1):
        CC_Port[t + 1] = CC_Port[t] * (1 + Port[t + 1])
        CC[t + 1] = CC[t] * (1 + mkt2[t + 1])
    print('Terminal wealth in Opt Port and Mkt  \n')
    print(CC_Port[T - 1])
    print(CC[T - 1])
    p = plt.plot(CC_Port)
    p1 = plt.plot(CC)
    plt.show()

# P_f_4()