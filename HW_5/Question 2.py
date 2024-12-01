import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

def analyze_with_n_15():
    """
    Analyze regression results when n=15, including OLS and LASSO
    """
    n = 15
    T = 100  # Sample size
    x = np.random.randn(T, n)  # Generate predictors

    # Set true coefficients
    b = np.zeros((n, 1))
    b[0] = 1  # First true coefficient
    b[1] = 2  # Second true coefficient
    b[2] = -2  # Third true coefficient

    # Generate response variable y
    y = np.matmul(x, b) + np.random.randn(T, 1) * 0.5
    y = y.flatten()  # Convert y to a 1D array

    # Ordinary Least Squares (OLS) regression
    reg = LinearRegression()
    reg.fit(x, y)

    print("OLS Results (n=15):")
    print("Intercept:", reg.intercept_)
    print("Coefficients:", reg.coef_)

    # LASSO regression
    lasso = Lasso(alpha=0.1)  # Alpha is the regularization strength
    lasso.fit(x, y)

    print("\nLASSO Results (n=15):")
    print("Intercept:", lasso.intercept_)
    print("Coefficients:", lasso.coef_)

def analyze_with_n_20():
    """
    Analyze regression results when n=20, including OLS and LASSO
    """
    n = 20
    T = 100  # Sample size
    x = np.random.randn(T, n)  # Generate predictors

    # Set true coefficients
    b = np.zeros((n, 1))
    b[0] = 1  # First true coefficient
    b[1] = 2  # Second true coefficient
    b[2] = -2  # Third true coefficient

    # Generate response variable y
    y = np.matmul(x, b) + np.random.randn(T, 1) * 0.5
    y = y.flatten()  # Convert y to a 1D array

    # Ordinary Least Squares (OLS) regression
    reg = LinearRegression()
    reg.fit(x, y)

    print("\nOLS Results (n=20):")
    print("Intercept:", reg.intercept_)
    print("Coefficients:", reg.coef_)

    # LASSO regression
    lasso = Lasso(alpha=0.1)  # Alpha is the regularization strength
    lasso.fit(x, y)

    print("\nLASSO Results (n=20):")
    print("Intercept:", lasso.intercept_)
    print("Coefficients:", lasso.coef_)

# Call functions to execute analyses for questions 1 and 2
print("Question 1: Regression analysis for n=15")
analyze_with_n_15()

print("\nQuestion 2: Regression analysis for n=20")
analyze_with_n_20()
