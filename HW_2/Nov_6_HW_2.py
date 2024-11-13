import pandas as pd                     # To load data, we use the package pandas
import numpy as np
import matplotlib.pyplot as plt
import cvxopt                 # install it first by running: "pip install cvxopt" via Spyder
from cvxopt import matrix, solvers
from numpy.ma.core import shape
from  Horizon_Tool.Data_Cal import *

df = pd.read_excel('../Data/Factors_July26_July11.xlsx')        # It has 5 columns:  date, mkt, size, b/m, riskree rate
df2 = pd.read_excel('../Data/Indu10_July26_July11.xlsx')       # return on 10 industry portfolios from Ken French's web

mkt = df.loc[:,"mkt"]/100                       # Mkt excess return;  divided by 100 due to data are in %
rf = df.loc[:,"rate"]/100
R1 = df2.loc[:, 'Indu1': ] / 100  # extract the 10 industry returns, R1 is T by 10.
R1 = np.array(R1)  # convert list to array to apply np.functions, T x 10
rf = np.array(rf)  # convert list to array to apply np.functions, a T-vector
# note: T, is not Tx1 in Python as the latter is 2-dim
T = len(df)  # The number of observations
Port = np.ones((T,))  # define this T-vector to store the returns on the portfolio
Re = np.ones((T, 10))  # creat storage for excess returns
for i in range(10):
    Re[:, i] = R1[:, i] - rf  # the excess return:  each indu substracts riskfree rate, Re[:,i]-rf
# print(shape(Re))
mu10 = np.mean(Re, axis=0)  # the mean taking each column of the matrix, a row vector of 10
# print(f"before {mu10}")
mu10 = mu10.T  # make it a column vector
V10 = np.cov(Re.T)  # the covariance estimate, 5 by 5
# print(V10)
VI = np.linalg.inv(V10)  # The inverse of V
# The optimal weights on the 5 risky aasets
gamma = 3  # The risk-averse coeff.
w10 = (1 / gamma) * VI @ mu10  # an alternative:  np.matmul(VI, mu10)
con_w = np.ones(1)
no_con_w = np.ones(1)

def q_one_1():
    global con_w, no_con_w
    # Compute the the Opt Port under bound constraints
    Q = gamma * V10  # the mapping of utility paramters into quadratic programming
    # print(f"previous {Q}")
    Q = matrix(Q)
    # print(Q)
    q = -mu10
    q = matrix(q)
    lb = np.ones((1, 10)) * 0.0  # lower bound
    ub = np.ones((1, 10)) * 0.3  # upper bound
    h = np.append(ub, lb, axis=1)
    h = h.T
    h = matrix(h)
    # print(h)
    G1 = np.eye(10)  # identiy matrix of order 10
    G2 = - np.eye(10)
    # print(G2)
    G = np.append(G1, G2, axis=0)
    G = matrix(G)
    solvers.options['show_progress'] = False  # this prevent print progress data
    sol = solvers.qp(Q, q, G, h)  # Format:  solvers.qp(Q, q, G, h, A, b)
    con_w = np.array(sol['x'])
    print('The Optimal wights on the 10 assets with constraints: 0 <=   <=.4 \n')
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    print(f"{con_w.T}")
    sol1 = solvers.qp(Q, q)  # using the solver for the unconstrained solution
    no_con_w = np.array(sol1['x'])
    print('\n The Optimal wights on the 10 assets with no constraints, using Solvers instead of formula  \n ')
    print(no_con_w.T)
    # print(type(no_con_w.T))
q_one_1()
# print(con_w.shape)
# con_w = con_w.reshape(-1)
# print(con_w.shape)

def q_one_2():
    global con_w
    # to be compatible with rf
    con_w = con_w.reshape(-1)
    # print(con_w)
    # print(Port.shape, con_w.shape, rf.shape)
    Port[0] = np.dot(con_w, Re[0]) + rf[0]  # return in the first period, the weight on rf is absorbed
    # into the previous excess return term, see formulas in the slides
    for t in range(T):
        Port[t] = np.dot(con_w, Re[t]) + rf[t]
    ExPort = Port - rf  # excess return of the optimla portfolio
    muP = ExPort.mean()
    sigP = calculate_sigma(ExPort)
    SharpeP = np.sqrt(12) * muP / sigP
    print(ExPort.shape, Port.shape, rf.shape)
    # print(ExPort.shape, Port.shape, rf.shape)  # double check the vectors are cpmpatible
    annualized_mean = (1 + muP) ** 12 - 1  # annualized mean
    annualized_std = sigP * np.sqrt(12)  # annualized std
    print("\nOptimal Portfolio Metrics:")
    print("Annualized Mean Return: {:.5f}".format(annualized_mean))
    print("Annualized Standard Deviation: {:.5f}".format(annualized_std))
    print("Annualized Sharpe Ratio: {:.5f}".format(SharpeP))
    print("-" * 50)

q_one_2()

def q_one_3():
    global no_con_w
    no_con_w = no_con_w.reshape(-1)
    for t in range(T):
        Port[t] = np.dot(no_con_w, Re[t]) + rf[t]
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
q_one_3()


def q_one_4():
    df = pd.read_excel('../Data/Factors_July26_July11.xlsx')
    mkt = df.loc[:, "mkt"] / 100 -rf
    mu = mkt.mean()  # The expected mkt excess return
    sig2 = mkt.var()  # The var of the mkt excess return
    sigma = np.sqrt(sig2)  # Its vol
    Sharpe = mu / sigma
    # Annualize the mean and standard deviation
    annualized_mean_mkt = (1 + mu) ** 12 - 1  # Annualized mean return
    annualized_std_mkt = sigma * np.sqrt(12)  # Annualized standard deviation
    Sharpe_mkt = np.sqrt(12) * mu / sigma  # Annualized Sharpe Ratio
    print("\nMarket Portfolio Metrics:")
    print("Annualized Mean Return: {:.5f}".format(annualized_mean_mkt))
    print("Annualized Standard Deviation: {:.5f}".format(annualized_std_mkt))
    print("Annualized Sharpe Ratio: {:.5f}".format(Sharpe_mkt))

q_one_4()

def q_one_5():
    global con_w,no_con_w
    # Compute the accu returns of the opt port and the mkt
    C_Port = np.ones((T,))
    no_con_w = no_con_w.reshape(-1)
    con_w = con_w.reshape(-1)
    # into the previous excess return term, see formulas in the slides

    for t in range(T):
        Port[t] = np.dot(no_con_w, Re[t]) + rf[t]
        C_Port[t] = np.dot(con_w, Re[t]) + rf[t]

    CC_Port = np.ones((T,))  # to store the accumulative returns
    CC_C_Port =  np.ones((T,))
    CC = np.ones((T,))  # to store the accumulative returns

    mkt2 = mkt + rf  # Add back riskfree rate to get pure mkt return

    CC_C_Port[0] = 1 + C_Port[0]
    CC_Port[0] = 1 + Port[0]  # initial accu return
    CC[0] = 1 + mkt2[0]  # initial accu return

    for t in range(T - 1):
        CC_Port[t + 1] = CC_Port[t] * (1 + Port[t + 1])
        CC[t + 1] = CC[t] * (1 + mkt2[t + 1])
        CC_C_Port[t + 1] = CC_C_Port[t] * (1 + C_Port[t + 1])
        # print(CC_C_Port[t],CC[t])

    print('Terminal wealth in Opt Port and Mkt  \n')
    p = plt.plot(CC_Port)
    p1 = plt.plot(CC)
    p2 = plt.plot(CC_C_Port)
    plt.ylim(0, 1e3)
    plt.show()

q_one_5()