import pandas as pd                     # To load data, we use the package pandas
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm                      # We use this package to do estimation
# %matplotlib inline
'''
Problem I):

Based on the program L7_PCA_Indu5.* (you have to make some changes) and the Indu10.* (the time length already matches), answer the following questions:
1)   What are the eigenvalues and eigenvectors of the sample covariance matrix ?

2)  Which are the coefficients for the first PCA?

3)   What is the R-squared of the regression of the industry returns on the first PCA?

4)   In comparison with regression on the market index, is the PCA better or worse in      terms of the value of the R-squared ?

5）(optional) Check how well the PCA factor works for your favorite stocks.
'''

df = pd.read_excel(
    '../Data/Factors_July26_May05.xlsx')  # It has 5 columns:  date, mkt, size, b/m, riskree rate                                                   # downloaded from Ken French's website
df2 = pd.read_excel('../Data/Indu10_July26_May05.xlsx')  # return on 5 industry portfolios from Ken French's web
mkt = df.loc[:, "mkt"] / 100  # Mkt excess return;  divided by 100 due to data are in %
rf = df.loc[:, "rate"] / 100
R1 = df2.loc[:, 'Indu1':] / 100  # extract the 5 industry returns, R1 is T by 5.
R1 = np.array(R1)  # convert list to array to apply np.functions
rf = np.array(rf)  # convert list to array to apply np.functions
T = len(df)  # The number of obs
Re = np.ones((T, 10))  # creat storage for excess returns
for i in range(10):
    Re[:, i] = R1[:, i] - rf  # the excess return:  each indu substracts riskfree rate, Re[:,i]-rf
V10 = np.cov(Re.T)  # the covariance estimate, 5 by 5
eigvals, eigvecs = np.linalg.eig(V10)  # use the np function
idx = eigvals.argsort()[::-1]  # *.argsort returns the indices
# one would use to sort the array
eigvals_sorted = eigvals[idx]
eigvecs_sorted = eigvecs[:, idx]

def P_O_O():
    print('The covariance matrix*10000 \n')
    print(V10*10000)                            # scaled it by 10^4 as the slides


    print('\nEigenvalues * 10000 (scaled):')
    print(np.round(eigvals * 10000, 2))

    print('\nCorresponding Eigenvectors (unsorted):')
    print(np.round(eigvecs, 4))

    print('\nSorted Eigenvalues * 10000 (scaled):')
    print(np.round(eigvals_sorted * 10000, 2))

    print('\nSorted Eigenvectors:')
    print(np.round(eigvecs_sorted, 4))
# P_O_O()

ones10 = np.ones((1, 10))

def P_O_T():
    A1 = eigvecs_sorted[:, 0]  # Coefficients for the first PCA, the 1st eigenvector
    print(f"Coefficients for the first PCA is \n{A1}")
    A2 = eigvecs_sorted[:, 1]  # that for the 2nd PCA; we will use here, just an example

    mu10 = np.mean(Re, axis=0) # the mean taking each column of the matrix, 1 by 5
    mu10 = mu10.reshape(1, 10) # make sure mu5 is 1 by 5
    onesT = np.ones((T, 1))  # T by 1
    RR = Re - onesT @ mu10  # de-mean the returns, T by 5

    #  the first PCA factor realizations, like the realized return on the market

    f = RR @ A1  # T by 1.  Alternative code:  np.matmul(RR,A1)
    #  Note the PCA analysis is usually applied to de-meaned data,
    #  PCA = A1*demeaned variables, so that the factor mean is zero.
    #  This has no impact on A1, etc.

    # Ratio of the first eigenvalue to the total


    Fraction = eigvals[0] / np.dot(ones10, eigvals)  # Remember sum of a vector is its dot with 1's

    print('The values of the first PCA factor in the first 3 periods ')
    print('             similar to the market factor in the first 3 periods \n')
    print(f[0:3])
    print('\n The fraction of the first eigenvalue relative to the total  \n')
    print(Fraction)


# P_O_T()
def P_O_Tr():

    # x = np.array(mkt)
    # x.shape = (T, 1)  # make sure the dimentionality
    A1 = eigvecs_sorted[:, 0]  # Coefficients of the first PCA
    mu10 = np.mean(Re, axis=0).reshape(1, 10)  # Mean returns (1x10)
    onesT = np.ones((T, 1))  # T x 1 matrix of ones
    RR = Re - onesT @ mu10  # De-mean the returns (T x 10)
    f = RR @ A1  # First PCA factor (T x 1)

    f = f.reshape(-1, 1)  # Ensure it is a column vector
    const = np.ones((T, 1))  # The constant part
    xx = np.hstack((const, f))  # Combine constant and PCA factor (T x 2)
    R2 = np.ones((10, 1))  # to store the R-squares
    for i in range(10):
        y = np.array(Re[:, i])  # The i-th excess asset return
        y.shape = (T, 1)
        reg = sm.OLS(endog=y, exog=xx)
        results = reg.fit()
        R2[i] = results.rsquared_adj
    # AvR2 = np.dot(ones10, R2) / 10
    print('The adjusted R^2 of the mkt factor on the first PCA \n')
    print(R2)
    return R2
    # print('  \n  the average \n')
    # print(AvR2)

R2 = P_O_Tr()

def P_O_F():

    x = np.array(mkt)
    x.shape = (T, 1)  # make sure the dimentionality
    const = np.ones((T, 1))
    xx = np.hstack((const, x))
    R2_market = np.ones((10, 1))
    for i in range(10):
        y = np.array(Re[:, i])
        reg = sm.OLS(endog=y, exog=xx)
        results = reg.fit()
        R2_market[i] = results.rsquared_adj
    AvR2_market = np.mean(R2_market)

    AvR2 = np.mean(R2)

    print("\nComparison of the average adjusted R^2 between the market factor and the PCA factor:")
    print(f"average adjusted R^2 market factor: {AvR2_market:.5f}, PCA factor: {AvR2:.5f}")
    if AvR2 > AvR2_market:
        print(
            "The average adjusted R^2 of the PCA factor is higher, indicating better explanatory power compared to the market factor.")
    elif AvR2 < AvR2_market:
        print(
            "The average adjusted R^2 of the market factor is higher, indicating better explanatory power compared to the PCA factor.")
    else:
        print(
            "The average adjusted R^2 of the market factor and the PCA factor are equal, indicating similar explanatory power.")

P_O_F()