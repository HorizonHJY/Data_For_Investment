import pandas as pd                     # To load data, we use the package pandas
import numpy as np
import matplotlib.pyplot as plt

import cvxopt                 # install it first by running: "pip install cvxopt" via Spyder
from cvxopt import matrix, solvers
from numpy.ma.core import shape

df = pd.read_excel('../Data/Factors_July26_July11.xlsx')        # It has 5 columns:  date, mkt, size, b/m, riskree rate
df2 = pd.read_excel('../Data/Indu10_July26_July11.xlsx')       # return on 10 industry portfolios from Ken French's web

mkt = df.loc[:,"mkt"]/100                       # Mkt excess return;  divided by 100 due to data are in %
rf = df.loc[:,"rate"]/100
R1 = df2.loc[:, 'Indu1': ] / 100  # extract the 10 industry returns, R1 is T by 10.
R1 = np.array(R1)  # convert list to array to apply np.functions, T x 10
rf = np.array(rf)  # convert list to array to apply np.functions, a T-vector
# note: T, is not Tx1 in Python as the latter is 2-dim
T = len(df)  # The number of observations
Re = np.ones((T, 10))  # creat storage for excess returns
for i in range(10):
    Re[:, i] = R1[:, i] - rf  # the excess return:  each indu substracts riskfree rate, Re[:,i]-rf
print(shape(Re))
mu10 = np.mean(Re, axis=0)  # the mean taking each column of the matrix, a row vector of 10
# print(f"before {mu10}")
mu10 = mu10.T  # make it a column vector
V10 = np.cov(Re.T)  # the covariance estimate, 5 by 5
# print(V10)
VI = np.linalg.inv(V10)  # The inverse of V
# The optimal weights on the 5 risky aasets
gamma = 3  # The risk-averse coeff.
w10 = (1 / gamma) * VI @ mu10  # an alternative:  np.matmul(VI, mu10)


def q_one_1():
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
    print(G2)
    G = np.append(G1, G2, axis=0)
    G = matrix(G)
    print(G1)
    print(G)
    solvers.options['show_progress'] = False  # this prevent print progress data
    sol = solvers.qp(Q, q, G, h)  # Format:  solvers.qp(Q, q, G, h, A, b)
    sola = np.array(sol['x'])
    print('The Optimal wights on the 10 assets with constraints: 0 <=   <=.4 \n')
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    print(sola.T)
    sol1 = solvers.qp(Q, q)  # using the solver for the unconstrained solution
    solb = np.array(sol1['x'])
    print('\n The Optimal wights on the 10 assets with no constraints, using Solvers instead of formula  \n ')
    print(solb.T)

q_one_1()