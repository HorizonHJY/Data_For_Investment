import pandas as pd  # To load data, we use the package pandas
# it is known as importing Modules, the packages we need
import numpy as np # To do mathematical operations, we use the package numpy
from datetime import datetime
import matplotlib.pyplot as plt  # To plot the returns data, we use the package matplotlib

# Load the monthly return of IBM, from 01/1934 to 12/2011
'''
Bases on the program L2_Analysis.ipynb and the data on IBM, S&P500 and riskfree rate from January 1934 to December 2011, 
answer the following questions over two sub-periods: Jan 1934 to December 2006, and January 2007 to December 2011
'''
filepath = "/Users/yuan_oli/Documents/Data Analysis for investment/HW/HW_1/Python1/"
ibm_name = 'IBM.xlsx'
sp_name ='SP500.xlsx'
risk_free_name = 'Riskfree.xlsx'
df_ibm = pd.read_excel(filepath+ibm_name)  # the data is in xlsx format
df_sp = pd.read_excel(filepath+sp_name)
df_rf = pd.read_excel(filepath+risk_free_name)
# print(df_rf)
Re_ibm = df_ibm.loc[:,"Return"]
Re_sp = df_sp.loc[:,"Return"]
#load the data for IBM
re_ibm_pf = df_ibm[(df_ibm["Date"] >= 19340131) & (df_ibm["Date"] <= 20061229)].loc[:,"Return"]
re_ibm_ps =  df_ibm[(df_ibm["Date"] >= 20070131) & (df_ibm["Date"] <= 20111230)].loc[:,"Return"] #class 'pandas.core.series.Series
#load the data for S&P 500
re_sp_pf = df_sp[(df_sp["Date"] >= 19340131) & (df_sp["Date"] <= 20061229)].loc[:,"Return"]
re_sp_ps =  df_sp[(df_sp["Date"] >= 20070131) & (df_sp["Date"] <= 20111230)].loc[:,"Return"]

def q_one(data_source,start_date, end_date,p_name):
    #change int to date formate for later outcome print.
    start_date = datetime.strptime(str(start_date), "%Y%m%d")
    end_date = datetime.strptime(str(end_date), "%Y%m%d")

    mu = np.mean(data_source)
    sig = np.var(data_source)  # The variance, i.e., the square of the standard deviation
    std = np.sqrt(sig)    # The standard deviation
    # print(f"{p_name} for the time period {start_date.date()} to {end_date.date()}:\nThe average monthly return is {mu:.5f} and standard deviation is {std:.5f}")
    return mu,std

# q_one(re_ibm_pf,19340131,20061229,"IBM")
# q_one(re_ibm_ps,20070131,20111230,"IBM")
# print('')
# q_one(re_sp_pf,19340131,20061229,"S&P")
# q_one(re_sp_ps,20070131,20111230,"S&P")

def sharp_ratio_cal(data_set,c_name,start_date,end_date):
    mu = np.mean(data_set)
    sig = np.var(data_set)# The variance, i.e., the square of the standard deviation
    std = np.sqrt(sig) # The standard deviation
    Sharpe_ratio = mu / std
    print(f'For {c_name} from {start_date} to {end_date}: \n'
          f'monthly Sharp Ratio is {Sharpe_ratio:.5f} and annulaized Sharpe ratio is {np.sqrt(12) * Sharpe_ratio:5f} ')
    return  Sharpe_ratio

def q_2():
    #load the risk-free rate for two time period
    df_rf_pf = df_rf[(df_rf["Date"] >= 193401) & (df_rf["Date"] <= 200612)].loc[:, "rate"]
    df_rf_ps = df_rf[(df_rf["Date"] >= 200701) & (df_rf["Date"] <= 201112)].loc[:, "rate"]
    # the excess return, i.e., return minus riskfree rate
    ER_SP_pf =  re_sp_pf - df_rf_pf / 100
    ER_SP_ps = re_sp_ps - df_rf_ps / 100
    ER_IMB_ps = re_ibm_pf - df_rf_pf / 100
    ER_IMB_pf= re_ibm_ps - df_rf_ps / 100
    #  divided by 100 b/c the rate data is in percentage points
    sharp_ratio_cal(ER_SP_pf,"S&P", "1934-01-31" , "2006-12-29")
    sharp_ratio_cal(ER_SP_ps,"S&P","2007-01-31","2011-12-30")
    print()
    sharp_ratio_cal(ER_IMB_ps,"IBM", "1934-01-31" , "2006-12-29")
    sharp_ratio_cal(ER_IMB_pf,"IBM","2007-01-31","2011-12-30")

def q_4(data_set,p_name):
    #initial the investment amount.
    ini_amo = 1000
    #Calculate the final
    for i in data_set:
        ini_amo = ini_amo * (1 + i)
    print(f"invest 1,000$ in {p_name} from  January 2007, at 2011-12-30 will be {ini_amo:.6f}")

#re_sp_ps: monthly return of S&P500 in Second Period :2007-01-31 to 2011-12-30.
#re_ibm_ps: monthly return of IBM in Second Period :2007-01-31 to 2011-12-30.
# q_4(re_sp_ps,'S&P500')
# q_4(re_ibm_ps,"IBM")

def q_5(data_set,start_date,end_date,p_name):
    skew = 0  # initialize it be zero
    kurt = 0
    start_date_5 = datetime.strptime(str(start_date), "%Y%m%d")
    end_date_5 = datetime.strptime(str(end_date), "%Y%m%d")

    T = len(data_set)  # Get the length, # of obvs (the headers of the Excel doesn't count)
    mu,sigma = q_one(data_set,start_date,end_date,p_name) #Call the question 1 part to get the sigma and mu for the input dataset

    for i in data_set:
        skew = skew + pow(i - mu, 3)  # sums the 3rd power terms successively
        kurt = kurt + pow(i - mu, 4)

    skew = (skew / pow(sigma, 3)) / T  # take the average
    kurt = (kurt / pow(sigma, 4)) / T

    print(f'\n{p_name} for the time period {start_date_5.date()} to {end_date_5.date()}:\n (Monthly) from skew is {skew:.6f}, kurt is {kurt:.6f} ')
    return skew,kurt

#call function to do calculation for each period
# q_5(re_ibm_pf,19340131,20061229,"IBM")
# q_5(re_ibm_ps,20070131,20111230,"IBM")
# print('')
# q_5(re_sp_pf,19340131,20061229,"S&P")
# q_5(re_sp_ps,20070131,20111230,"S&P")

import seaborn as sns
def q_6():
    plt.figure(figsize=(10, 6))
    sns.kdeplot(re_ibm_ps, color='green', linewidth=2)
    # sns.histplot(re_sp_pf,bins=120, kde=True, color='green')
    # plt.title("Distribution of Monthly Returns for S&P 500 from 1934-01-31 to 2006-12-29")
    plt.title("Distribution of Monthly Returns for IBM from 2007-01-31 to 2011-12-30")
    plt.xlabel("Monthly Returns")
    plt.ylabel("Frequency")
    plt.show()

# S&P 500
def q_7(data_set,name):
    max_re = max(data_set)
    min_re = min(data_set)
    print(f"For {name} the maximum monthly return is {max_re} and Lowest return is {min_re}\n")

# q_7(Re_sp,"S&P")
# q_7(Re_ibm,"IBM")

def q_8(data_set,p_name):
    #create a copy to store change
    data_set_new = data_set.copy()
    #get the 10% of the top best months
    threshold = data_set.quantile(0.9)
    #set the top 10% return as 0
    data_set_new[data_set_new >= threshold] = 0
    #call q_4 to calculate new result
    q_4(data_set_new,p_name)

q_8(re_sp_ps,"S&P 500")
q_8(re_ibm_ps,"IBM")
