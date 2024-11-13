import pandas as pd                     # To load data, we use the package pandas
import numpy as np
import scipy.stats as si
import scipy.linalg as la

np.random.seed(888)
#Question 1
def P_o_o(T):
    mu = 0.1
    sigma = np.sqrt(.2)
    ini_matrix = np.random.randn(T)
    result_o_o = mu + sigma * ini_matrix
    result_o_o.mean()
    var1 = np.var(result_o_o)
    std1 = np.sqrt(var1)
    print(f"sample mean is {result_o_o.mean():.5f} and sample variance {std1:.6f}")
# P_o_o(60)
#Question 2
# P_o_o(600)
# P_o_o(6000)

#Question 3
def BS_call(S, X, T, r, sigma):
    # S: spot price; X: strike; T: time to maturity; r: riskfree rate; sigma: volatility
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N1 = si.norm.cdf(d1)
    N2 = si.norm.cdf(d2)
    call = S * N1 - X * np.exp(-r * T) * N2
    return call

def P_o_s(sim_num):
    S, X, T, r, sigma = 50, 50, 0.25, 0.1, 0.3
    bs_price = BS_call(S, X, T, r, sigma)
    print("Black-Scholes price:", bs_price)
    M = sim_num
    muT = (r - 0.5 * sigma ** 2) * T
    scalez = sigma * np.sqrt(T)
    z = np.random.randn(M, 1)
    ST = S * np.exp(muT + scalez * z)
    payoffs = np.maximum(ST - X, 0)
    mc_price = np.exp(-r * T) * np.mean(payoffs)
    print("Simulation Price:", mc_price)

# P_o_s(1000)
# P_o_s(1000000)

def P_o_t():
    mean = [0.1, 0.15]
    std_dev = [0.2, 0.3]
    correlation = 0.2
    cov_matrix = [[std_dev[0]**2, correlation * std_dev[0] * std_dev[1]],
                  [correlation * std_dev[0] * std_dev[1], std_dev[1]**2]]

    samples_60 = np.random.multivariate_normal(mean, cov_matrix, 60)
    samples_600 = np.random.multivariate_normal(mean, cov_matrix, 600)

    print("60 sample：\n", samples_60)
    print("600 sample：\n", samples_600)

# P_o_t()

def P_o_f():
    df_tsla= pd.read_excel("../Data/TSLA.xlsx")
    alpha = 0.05
    T = len(df_tsla['Return'])
    df = T - 1

    t95 = si.t.ppf(1 - alpha / 2, df)
    sample_mean = np.mean(df_tsla['Return'])
    sample_std = np.std(df_tsla['Return'], ddof=1)
    margin_of_error = t95 * (sample_std / np.sqrt(T))
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error
    # print(df_tsla.head(10))
    print(f"95% CI: [{lower_bound:.5f}, {upper_bound:.5f}]")

# P_o_f()

def P_o_s():
    df_tsla = pd.read_excel("../Data/TSLA.xlsx")
    data = df_tsla['Return']
    n_bootstraps = 1000
    boot_means = []
    for _ in range(n_bootstraps):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(bootstrap_sample))
    lower_bound = np.percentile(boot_means, 2.5)
    upper_bound = np.percentile(boot_means, 97.5)

    print(f"95% CI Using bootstrap method: [{lower_bound}, {upper_bound}]")

# P_o_s()
# we need the package to compute the Cholesky decomposition
N = 2  # Updated dimension
T = 30

mu0 = np.array([[1.3, 1.0]]).T  # True mean, updated for N=2
V0 = np.identity(N)  # set the covariance matrix to the identity matrix

L1 = la.cholesky(V0)  # Cholesky decomposition: V0 = L1'*L1
L = L1.T  # no need of doing this for identity matrix

M = 15  # Updated number of draws or simulations
R = np.ones((T, N))  # R to store the T returns, data set in each draw; different across draws
Z1 = np.ones((M, 1))  # M by 1 vector, to store the errors in each draw, and save them all
Z2 = np.ones((M, 1))
Z3 = np.ones((M, 1))

for i in range(M):  # this loop runs the simulation

    for t in range(T):  # generate the data of length T
        e = np.random.randn(N, 1)
        Y = mu0 + np.matmul(L, e)
        R[t, :] = Y.T

    # sample estimation first, and compute the variables needed later
    muR = np.mean(R, axis=0)
    muR = muR.T  # N by 1
    V = np.cov(R.T)  # the covariance estimate, N by N
    VI = np.linalg.inv(V)  # The inverse of V

    eigvals, eigvecs = np.linalg.eig(V)  # get the eigenvalues and eigenvectors

    lbar = np.mean(eigvals)
    lambda1 = np.max(eigvals)

    # The first shrinkage estimator
    b1 = np.mean(muR, axis=0) * np.ones((N, 1))
    muR.shape = (N, 1)  # make sure N by 1
    a1 = muR - b1
    a = np.dot(a1.T, a1)  # The term (muR-b1)'*(muR-b1)
    alpha = (1 / T) * (N * lbar - 2 * lambda1) / a
    Smu1 = (1 - alpha) * muR + alpha * b1

    # The second shrinkage estimator
    onesN = np.ones((N, 1))
    B = np.matmul(onesN.T, VI)
    b11 = np.matrix(onesN.T) * np.matrix(VI) * np.matrix(muR)
    b12 = np.matrix(onesN.T) * np.matrix(VI) * np.matrix(onesN)
    c11 = b11.item()  # make an array to a scalar!
    c12 = b12.item()
    b2 = (c11 / c12) * onesN
    b = np.dot((muR - b2).T, (muR - b2))  # The term (muR-b2)'*(muR-b2)
    alpha = (1 / T) * (N * lbar - 2 * lambda1) / b
    Smu2 = (1 - alpha) * muR + alpha * b2

    # store the squared errors of the estimates
    Z1[i] = np.dot((muR - mu0).T, (muR - mu0))  # (muR-mu0)'*(muR-mu0)
    Z2[i] = np.dot((Smu1 - mu0).T, (Smu1 - mu0))
    Z3[i] = np.dot((Smu2 - mu0).T, (Smu2 - mu0))

Err1 = np.sqrt(np.mean(Z1))
Err2 = np.sqrt(np.mean(Z2))
Err3 = np.sqrt(np.mean(Z3))

print('  The mean squared errors of estimating the true mean  ')
print('      based on sample mean, Shrinkage 1 & 2   \n')
print(Err1, Err2, Err3)
