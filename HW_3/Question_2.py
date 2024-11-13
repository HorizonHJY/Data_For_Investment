import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import f

factors_df = pd.read_excel('../Data/Factors_July26_July11.xlsx')
returns_df = pd.read_excel('../Data/Indu10_July26_July11.xlsx')
# print(returns_df.head())
mkt = factors_df['mkt'] / 100    # Mkt excess return;  divided by 100 due to data are in %
rf = factors_df['rate'] / 100
R1 = returns_df.loc[:, 'Indu1':'Indu10'] / 100  # extract the 10 returns, R1 is T by 10.

R1 = np.array(R1)    # convert list to array to apply np.functions
rf = np.array(rf)    # convert list to array to apply np.functions
T = len(factors_df)  # Number of observations
Re = np.ones((T, 10))    # creat storage for excess returns

for i in range(10):
    Re[:, i] = R1[:, i] - rf   # the excess return:  each indu substracts riskfree rate, Re[:,i]-rf

# The Sharpe ratio of the mkt, needed later for computing the GRS test

mu = mkt.mean()  # The expected mkt excess return
sig2 = mkt.var()  # The var of the mkt excess return
sigma = np.sqrt(sig2)  # Its vol
Sharpe = mu / sigma

# run regression of each asset on the mkt

x = np.array(mkt).reshape(T, 1)  # Market excess returns，make sure the dimentionality
const = np.ones((T, 1))  # The constant part
xx = np.hstack((const, x))    # Add the constant part to x


coeff = np.ones((10, 2))  # Store alpha and beta
tRatio = np.ones((10, 2))  # to store all the t-ratios

for i in range(10):
    y = np.array(Re[:, i]).reshape(T, 1)  # The i-th excess asset return
    reg = sm.OLS(endog=y, exog=xx)
    results = reg.fit()
    coeff[i, :] = results.params  # paramter estimates, output of sm.OLS
    tRatio[i, :] = results.tvalues   # t-ratios or t-values, output of sm.OLS


alphas = coeff[:, 0]
print("Alphas of the CAPM regression:")
print(alphas)

# 2. Two largest alphas by absolute value
largest_alphas_indices = np.argsort(np.abs(alphas))[-2:][::-1]
print("\nThe two largest alphas in terms of absolute values:")
for idx in largest_alphas_indices:
    print(f"Portfolio {idx + 1}: Alpha = {alphas[idx]:.5f}")

# 3. Explain why those alphas are hard to explain (manual step, context-specific)
# Example explanation can be added in comments or separate analysis outside the code.

# 4. Test if CAPM is rejected by the data
Alphas = coeff[:, 0].reshape(10, 1)
Betas = coeff[:, 1]

E = Re - xx @ coeff.T  # The residuals
Sigma = E.T @ E / T  # The sample covariance matrix of the residuals
SI = np.linalg.inv(Sigma)    # The inverse of Sigma

Q1 = Alphas.T @ SI @ Alphas   #  Q1 = Alpha'*Sigma^(-1)*Alpha = Alpha'*Qw

Q = Q1.item()  # make an array to a scalar !

N = 10
Theta2 = (mkt.mean() / mkt.std())**2  # Sharpe ratio squared
GRS = ((T - N - 1) / N) * Q / (1 + Theta2)


c = f.cdf(GRS, N, T - N - 1)  # This compute the F-distribution, Prob ( x < GRS )
GRSp = 1 - c  #  P-value of the test decided by the F-distribution

# Output results
print('The quadratic of the alphas (pricing errors):\n')
print(Q)
print('\nThe GRS test statistic:\n')
print(GRS)
print('\nThe p-value:\n')
print(GRSp)

# Interpret CAPM rejection
if GRSp < 0.05:
    print("\nCAPM is rejected at the 5% significance level.")
else:
    print("\nCAPM is not rejected at the 5% significance level.")
