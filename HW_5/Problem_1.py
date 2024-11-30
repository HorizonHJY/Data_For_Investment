import pandas as pd  # To load data, we use the package pandas
import numpy as np
import statsmodels.api as sm  # We use this package to do statical estimation
import statsmodels.formula.api as smf

# Load the data and define variables, 1926:12-2010:12

df = pd.read_excel('../Data/Returns_handbook_Python_data.xlsx', 'Monthly')  # Monthly is the name of Sheet1

# Market risk premium
market_return = np.array(df.loc[672: 1679, "CRSP S&P 500 value-weighted return with dividends"])
# 671th row is 1927:01 in the data
# Excel row is 673 bc/ Python does not count header and starts from 0
r_f_lag = np.array(df.loc[671: 1678, "Risk-free rate"])
equity_premium = market_return - r_f_lag  # excess return
  # Predictors

D12 = np.array(df.loc[671 : 1678,   "12-month moving sum of S&P 500 dividends"]); # dividends
SP500 = np.array(df.loc[671 : 1678, "S&P 500 index"]);
DP = np.log(D12) - np.log(SP500);                                # log dividend-price ratio
SP500_lag = np.array(df.loc[671 : 1678, "S&P 500 index"]);   # S&P 500 index, lagged (1926:12-2010:11)
DY = np.log(D12) - np.log(SP500_lag);       # log dividend yield
E12 = np.array(df.loc[671 : 1678,      "12-month moving sum of S&P 500 earnings"]); # earnings
EP = np.log(E12) - np.log(SP500);       # log earnings-price ratio
DE = np.log(D12) - np.log(E12);        # log dividend-payout ratio
SVAR = np.array(df.loc[671 : 1678, "Monthly sum of squared daily returns on S&P 500 index"]); # volatility
BM = np.array(df.loc[671 : 1678, "DJIA book-to-market value ratio"]);   # book-to-market ratio
NTIS = np.array(df.loc[671 : 1678, "Net equity expansion"]); # net equity issuing activity
TBL = np.array(df.loc[671 : 1678, "3-month Treasury bill yield (secondary market)"]); # T-bill rate
LTY = np.array(df.loc[671 : 1678, "Long-term government bond yield"]); # long-term government bond yield
LTR = np.array(df.loc[671 : 1678, "Long-term government bond return"]); # long-term government bond return
TMS = LTY - TBL; # term spread
AAA = np.array(df.loc[671 : 1678, "Moodys AAA-rated corporate bond yield"]); # AAA-rated corporate bond yield
BAA = np.array(df.loc[671 : 1678, "Moodys BAA-rated corporate bond yield"]); # BAA-rated corporate bond yield
DFY = BAA - AAA; # default yield spread
CORPR = np.array(df.loc[671 : 1678, "Long-term corporate bond return"]); # long-term corporate bond return
DFR = CORPR - LTR; # default return spread
INFL_lag = np.array(df.loc[671 : 1678, "CPI (all urban consumers) inflation rate"]); # inflation, lagged (1926:12-2010:11)
N = 14;
T = 1008
ECON = np.vstack((DP, DY, EP, DE, SVAR, BM, NTIS, TBL, LTY, LTR, TMS, DFY, DFR, INFL_lag)).T;
# print(ECON.shape), one should get (1008* 14)

y = np.array(equity_premium)
y.shape = (T, 1)  # make sure the dimentionality

onesT = np.ones((T, 1))  # The constant part

coeff = np.ones((N, 2))  # to store all the alphas and betas
tValues = np.ones((N, 2))
R2a = np.ones((N, 1))

for i in range(N):
    x = ECON[:, i]  # i-th predictor
    x = np.array(x)
    x.shape = (T, 1)
    xx = np.hstack((onesT, x))  # add the constant part to x
    reg = sm.OLS(endog=y, exog=xx)
    results = reg.fit()
    coeff[i, :] = results.params  # paramter estimates, output of sm.OLS
    tValues[i, :] = results.tvalues
    R2a[i] = results.rsquared_adj

slope = coeff[:, 1].reshape(-1, 1)  # another to make it N by 1 vector
slopeTvalue = tValues[:, 1].reshape(-1, 1)

Output = np.hstack((slope, slopeTvalue, R2a))
print(' Slope, t-value, Adjusted R-sqaured   \n')
print(Output)