import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import ElasticNet



def l_10_q_1():
    n = 15  # the number of x's
    T = 100  # sample size
    x = np.random.randn(T, n)  # Generate x's from N(0,1)
    b = np.zeros((n, 1))  # To store the coefficients
    b[0] = 1  # First true coefficient
    b[1] = 2  # Second true coefficient
    b[2] = -2  # Third true coefficient
    y = 0 + np.matmul(x, b) + np.random.randn(T, 1) * 0.5  # the simulated y's
    y.shape = (T,)  # a colum vector of 100 by 1
    #  Traditional OLS regression analysis
    reg = LinearRegression()  # shorthand the regression function
    reg.fit(x, y)  # runs a traditional regression of y on x
    print(reg.intercept_)  # the intercept
    print(reg.coef_)  # the slopes
    alpha = 0.5
    lesso = linear_model.Lasso(alpha)
    lesso.fit(x, y)
    print('Lesso result:')
    print(lesso.intercept_)  # the intercept
    print(lesso.coef_)  # the slopes

def l_10_q_2():
    n = 20  # the number of x's
    T = 100  # sample size
    x = np.random.randn(T, n)  # Generate x's from N(0,1)
    b = np.zeros((n, 1))  # To store the coefficients
    b[0] = 1  # First true coefficient
    b[1] = 2  # Second true coefficient
    b[2] = -2  # Third true coefficient
    y = 0 + np.matmul(x, b) + np.random.randn(T, 1) * 0.5  # the simulated y's
    y.shape = (T,)  # a colum vector of 100 by 1
    #  Traditional OLS regression analysis
    reg = LinearRegression()  # shorthand the regression function
    reg.fit(x, y)  # runs a traditional regression of y on x
    print(reg.intercept_)  # the intercept
    print(reg.coef_)  # the slopes
    alpha = 0.5
    lesso = linear_model.Lasso(alpha)
    lesso.fit(x, y)
    print('lesso result:')
    print(lesso.intercept_)  # the intercept
    print(lesso.coef_)  # the slopes


def l_10_q_3_1():
    n = 15  # the number of x's
    T = 100  # sample size
    x = np.random.randn(T, n)  # Generate x's from N(0,1)
    b = np.zeros((n, 1))  # To store the coefficients
    b[0] = 1  # First true coefficient
    b[1] = 2  # Second true coefficient
    b[2] = -2  # Third true coefficient
    y = 0 + np.matmul(x, b) + np.random.randn(T, 1) * 0.5  # the simulated y's
    y.shape = (T,)  # a colum vector of 100 by 1
    #  Traditional OLS regression analysis
    reg = LinearRegression()  # shorthand the regression function
    reg.fit(x, y)  # runs a traditional regression of y on x
    print(reg.intercept_)  # the intercept
    print(reg.coef_)  # the slopes
    alpha = 0.5
    ridge = linear_model.Ridge(alpha)
    ridge.fit(x, y)
    print('Ridge result:')
    print(ridge.intercept_)  # the intercept
    print(ridge.coef_)  # the slopes

def l_10_q_3_2():
    n = 20  # the number of x's
    T = 100  # sample size
    x = np.random.randn(T, n)  # Generate x's from N(0,1)
    b = np.zeros((n, 1))  # To store the coefficients
    b[0] = 1  # First true coefficient
    b[1] = 2  # Second true coefficient
    b[2] = -2  # Third true coefficient
    y = 0 + np.matmul(x, b) + np.random.randn(T, 1) * 0.5  # the simulated y's
    y.shape = (T,)  # a colum vector of 100 by 1
    #  Traditional OLS regression analysis
    reg = LinearRegression()  # shorthand the regression function
    reg.fit(x, y)  # runs a traditional regression of y on x
    print(reg.intercept_)  # the intercept
    print(reg.coef_)  # the slopes
    alpha = 0.5
    ridge = linear_model.Ridge(alpha)
    ridge.fit(x, y)
    print('Ridge result:')
    print(ridge.intercept_)  # the intercept
    print(ridge.coef_)  # the slopes

def l_10_q_4_1():
    n = 15  # the number of x's
    T = 100  # sample size
    x = np.random.randn(T, n)  # Generate x's from N(0,1)
    b = np.zeros((n, 1))  # To store the coefficients
    b[0] = 1  # First true coefficient
    b[1] = 2  # Second true coefficient
    b[2] = -2  # Third true coefficient

    y = 0 + np.matmul(x, b) + np.random.randn(T, 1) * 0.5  # the simulated y's
    y.shape = (T,)  # a colum vector of 100 by 1
    #  Traditional OLS regression analysis
    reg = LinearRegression()  # shorthand the regression function
    reg.fit(x, y)  # runs a traditional regression of y on x
    print(reg.intercept_)  # the intercept
    print(reg.coef_)  # the slopes
    l1_ratio = 0.5
    alpha = 0.5
    enet = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    enet.fit(x, y)
    print('ElasticNet result:')
    print(enet.intercept_)  # the intercept
    print(enet.coef_)  # the slopes


def l_10_q_4_2():
    n = 20  # the number of x's
    T = 100  # sample size
    x = np.random.randn(T, n)  # Generate x's from N(0,1)
    b = np.zeros((n, 1))  # To store the coefficients
    b[0] = 1  # First true coefficient
    b[1] = 2  # Second true coefficient
    b[2] = -2  # Third true coefficient

    y = 0 + np.matmul(x, b) + np.random.randn(T, 1) * 0.5  # the simulated y's
    y.shape = (T,)  # a colum vector of 100 by 1
    #  Traditional OLS regression analysis
    reg = LinearRegression()  # shorthand the regression function
    reg.fit(x, y)  # runs a traditional regression of y on x
    print(reg.intercept_)  # the intercept
    print(reg.coef_)  # the slopes
    l1_ratio = 0.5
    alpha = 0.5
    enet = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    enet.fit(x, y)
    print('ElasticNet result:')
    print(enet.intercept_)  # the intercept
    print(enet.coef_)  # the slopes




np.random.seed(888)
# l_10_q_1()
# l_10_q_2()
# l_10_q_3_1()
# l_10_q_3_2()
l_10_q_4_1()
# l_10_q_4_2()