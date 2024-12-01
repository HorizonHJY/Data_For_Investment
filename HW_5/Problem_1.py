import pandas as pd
import numpy as np
import statsmodels.api as sm

def regression_analysis(y, x):
    x_with_const = sm.add_constant(x)  # Add constant term
    reg = sm.OLS(y, x_with_const)
    results = reg.fit()

    # Extract regression coefficients and significance
    beta = results.params[1]
    p_value = results.pvalues[1]
    t_value = results.tvalues[1]

    return results, beta, p_value, t_value


def analyze_significance(beta, p_value, t_value, alpha=0.05):
    print(f"Beta: {beta:.10f}")
    print(f"T-value: {t_value:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < alpha:
        print(f"The Beta coefficient is significant at the {alpha*100:.0f}% level (p-value: {p_value:.4f}).")
    else:
        print(f"The Beta coefficient is NOT significant at the {alpha*100:.0f}% level (p-value: {p_value:.4f}).")


def question_1(merged_df):
    """
    Question 1: Regress current stock returns Rt on market returns Rmkt,t
    """
    stock_excess_return = merged_df['Return'] - merged_df['3-month Treasury bill yield (secondary market)']
    market_excess_return = merged_df['S&P 500 index'] - merged_df['3-month Treasury bill yield (secondary market)']
    y = stock_excess_return.values.reshape(-1, 1)
    x = market_excess_return.values.reshape(-1, 1)

    results, beta, p_value, t_value = regression_analysis(y, x)

    print("\nQuestion 1: Regression Results (Rt on Rmkt,t):")
    print(results.summary())
    analyze_significance(beta, p_value, t_value)


def question_2(merged_df):
    """
    Question 2: Regress current stock returns Rt on lagged market returns Rmkt,t-1
    """
    merged_df['Market_Excess_Return_Lag'] = merged_df['S&P 500 index'].shift(1) - merged_df[
        '3-month Treasury bill yield (secondary market)']
    merged_df = merged_df.dropna()  # Remove missing values caused by lagging

    stock_excess_return = merged_df['Return'] - merged_df['3-month Treasury bill yield (secondary market)']
    market_excess_return_lag = merged_df['Market_Excess_Return_Lag']
    y = stock_excess_return.values.reshape(-1, 1)
    x = market_excess_return_lag.values.reshape(-1, 1)

    results, beta, p_value, t_value = regression_analysis(y, x)

    print("\nQuestion 2: Regression Results (Rt on Rmkt,t-1):")
    print(results.summary())
    analyze_significance(beta, p_value, t_value)


def question_3(merged_df):
    """
    Question 3: Regress current stock returns Rt on lagged stock returns Rt-1
    """
    merged_df['Stock_Excess_Return_Lag'] = merged_df['Return'].shift(1) - merged_df[
        '3-month Treasury bill yield (secondary market)']
    merged_df = merged_df.dropna()  # Remove missing values caused by lagging

    stock_excess_return = merged_df['Return'] - merged_df['3-month Treasury bill yield (secondary market)']
    stock_excess_return_lag = merged_df['Stock_Excess_Return_Lag']
    y = stock_excess_return.values.reshape(-1, 1)
    x = stock_excess_return_lag.values.reshape(-1, 1)

    results, beta, p_value, t_value = regression_analysis(y, x)

    print("\nQuestion 3: Regression Results (Rt on Rt-1):")
    print(results.summary())
    analyze_significance(beta, p_value, t_value)


# Load data
ibm_df = pd.read_excel('../Data/IBM.xlsx')
market_df = pd.read_excel('../Data/Returns_handbook_Python_data.xlsx', sheet_name='Monthly')

# Merge data
ibm_df['YearMonth'] = ibm_df['Date'].astype(str).str[:6].astype(int)
market_df['YearMonth'] = market_df['Date (yyyymm)']
merged_df = pd.merge(ibm_df, market_df, on='YearMonth', how='inner')

# Run the questions
# question_1(merged_df)
# question_2(merged_df)
question_3(merged_df)
