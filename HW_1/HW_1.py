'''
a)Type 100*(1+0.15) at the console (bottom right), and hit Enter;
b)Type r=.15, Enter;
c)Type 100*pow(1+r,1), Enter;   [you get the value 100*(1+r)]
d)Type 100*pow(1+r,10), Enter;   [you get the value 100*(1+r)^10]
e)Type 100*pow(1+r,-2), Enter;   [you get the value 100/(1+r)^2]
f)How do you compute the present value of 10 payments:
              100/(1+r) + 100/(1+r)^2+…+ 100/(1+r)^10, enter return.
 (Hint:  set sum=0;  use a loop:    for i in range(10): sum=sum + ?)
'''
from fileinput import filename


#a
# print(100*(1+0.15))
#
def qf():
    sum = 0
    for i in range(1,11):
        sum += 100*pow(1+.15,i)
    print(f"The result for '100/(1+r) + 100/(1+r)^2+…+ 100/(1+r)^10' is {sum}")

import numpy as np
def q_t():
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    R = np.array([.10,-.05,.20])
    port = np.dot (w,R)
    print(port)

import pandas as pd
def cal_er(indices,datasource):
    sum_return_1 = 0
    for i in indices:
        if datasource[i] >0:
            pass
        else:
            datasource.iloc[i] = 0
        sum_return_1 += datasource[i]
    average_return_1 = sum_return_1 / len(indices)
    return  average_return_1

def q_f():
    filepath = "/Users/yuan_oli/Documents/Data Analysis for investment/HW/HW_1/Python1/"
    data_name = "IBM.csv"
    df = pd.read_csv(filepath + data_name)  # note: To write a data to cvs
    Re = df.loc[:, "Return"]
    indices = df[(df["Date"] > 19340131) & (df["Date"] < 20061229)].index.tolist()
    indices_2 = df[(df["Date"] > 20070131) & (df["Date"] < 20111230)].index.tolist()
    ave_r1 = cal_er(indices,Re)
    ave_r2 = cal_er(indices_2,Re)
    print(f"The average return from Jan 1934 to December 2006 to is {ave_r1*100:.2f}% :")
    print(f"The average return from Jan 2007 to December 2011 to is {ave_r2*100:.2f}% :")

