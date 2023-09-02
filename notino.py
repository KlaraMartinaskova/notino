# Notino Test Assigment
# Author: Klara Martinaskova
# Task: Evaluate an AB test of the recommendation algorithm

#!/usr/bin/env python3.9
# coding=utf-8

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
import pingouin as pg
import zipfile
import os
import csv


# # open CSV files
# with open('clients_final.csv', 'r') as file:
#     csv_clients = csv.reader(file)
# with open('orders_final.csv', 'r') as file:
#     csv_orders = csv.reader(file)

# open csv file ans save as dataframe
df_clients = pd.read_csv('clients_final.csv',encoding="cp1250", sep=",", low_memory=False)
df_orders = pd.read_csv('orders_final.csv',encoding="cp1250", sep=",", low_memory=False)

# Analysis of abUser column
abUser_unique = df_clients['abUser'].unique() # 99 is not present here (just NaN)
txt = "Unique values in abUser: {}"
print(txt.format(abUser_unique))


# •	What about the users with an unassigned group? Bambino thinks the test is fine if their share is below 0.5%. 
reco_count = df_clients['abUser'][df_clients['abUser']==1].count()
control_count = df_clients['abUser'][df_clients['abUser']==2].count()
other_count = (df_clients['abUser']).isna().sum()

percent = other_count/((reco_count + control_count + other_count))*100


if percent < 0.5:
    print("The test is fine, share is below 0.5%.")
    print("Percent of users with an unassigned group: {:.2f}%".format(percent))
else:
    print("The test is not fine, share is {:.2f}%".format(percent))

# •	What about the orders that are not in GA data? What is their share? How do you propose to handle them?
df_join = df_clients.merge(df_orders, on='orderNumber', how='inner') # inner join
df_join_count = df_join['orderNumber'].count() # number of common orders in both dataframes
df_clients_count = df_clients['orderNumber'].count() # number of orders in df_clients
share = 1-(df_join_count/df_clients_count) # share of orders that are not in GA data

print("Share of orders that are not in GA data: {:.2f}%".format(share*100))

#•	Does the “reco group” earn, on average, a greater revenue? Does it have larger orders? 
# Propose appropriate metrics and visualize them. Is there any other metric you may wish to evaluate?

# reco group =  abUser == 1
# control group = abUser == 2

reco_mean = df_join['revenue'][df_join['abUser']==1].mean() 
control_mean = df_join['revenue'][df_join['abUser']==2].mean()

# plot relationship between quantity and revenue
plt.scatter(df_join['quantity'][df_join['abUser']==1], df_join['revenue'][df_join['abUser']==1], alpha=0.5, marker='o')
plt.scatter(df_join['quantity'][df_join['abUser']==2], df_join['revenue'][df_join['abUser']==2], alpha=0.5, marker = "x")
plt.xlabel('Quantity')
plt.ylabel('Revenue')
plt.legend(['reco group', 'control group'])
plt.show()

### detected outliner, which does not change the mean quantity result

reco_max = df_join['revenue'][df_join['abUser']==1].idxmax()

df_join_without_outliers = df_join.drop(reco_max)

plt.scatter(df_join_without_outliers['quantity'][df_join_without_outliers['abUser']==1], df_join_without_outliers['revenue'][df_join_without_outliers['abUser']==1], alpha=0.5, marker='o')
plt.scatter(df_join_without_outliers['quantity'][df_join_without_outliers['abUser']==2], df_join_without_outliers['revenue'][df_join_without_outliers['abUser']==2], alpha=0.5, marker = "x")
plt.xlabel('Quantity')
plt.ylabel('Revenue')
plt.legend(['reco group', 'control group'])
plt.show()

# T-test for independent samples unpair
# For data without outliers

# from https://www.geeksforgeeks.org/how-to-perform-an-f-test-in-python/

def f_test(group1, group2):
    f = np.var(group1, ddof=1)/np.var(group2, ddof=1)
    nun = group1.size-1
    dun = group2.size-1
    p_value = 1-stats.f.cdf(f, nun, dun)
    return f, p_value

### For revenue
print("T-test for revenue without outliners: ")
print("")

data_reco_without_outliers1 = df_join_without_outliers['revenue'][df_join_without_outliers['abUser']==1]
data_control_without_outliers1 = df_join_without_outliers['revenue'][df_join_without_outliers['abUser']==2]

# Shapiro-wilk test - Normality test:
test_normality_reco_revenue = stats.shapiro(data_reco_without_outliers1 )
test_normality_control_revenue = stats.shapiro(data_control_without_outliers1)
print("Shapiro-wilk test - Normality test: ")
print(test_normality_reco_revenue)
print(test_normality_control_revenue)
print("")

# perform F-test - Homogeneity test
reco_ftest = f_test(data_reco_without_outliers1, data_control_without_outliers1)
print("F-test - Homogeneity test: ")
print(reco_ftest)
print("")

# Two ways to calculate t-test

result_stats = stats.ttest_ind(a=data_reco_without_outliers1, b=data_control_without_outliers1 , equal_var=True)
print(result_stats)
print("")

result = pg.ttest(data_reco_without_outliers1,
                  data_control_without_outliers1,
                  correction=True)
print(result)
print("---------------------------------")

### For quantity
print("T-test for quantity without outliners: ")
print("")
data_reco_without_outliers2 = df_join_without_outliers['quantity'][df_join_without_outliers['abUser']==1]
data_control_without_outliers2 = df_join_without_outliers['quantity'][df_join_without_outliers['abUser']==2]

# Shapiro-wilk test - Normality test:
test_normality_reco_quantity = stats.shapiro(data_reco_without_outliers2)
test_normality_control_quantity = stats.shapiro(data_control_without_outliers2)
print("Shapiro-wilk test - Normality test: ")
print(test_normality_reco_quantity)
print(test_normality_control_quantity)
print("")

# perdom F test - Homogeneity test
control_ftest = f_test(data_reco_without_outliers2,data_control_without_outliers2)
print("F-test - Homogeneity test: ")
print(control_ftest)
print("")

# Two ways to calculate t-test
result_stats2 = stats.ttest_ind(a=data_reco_without_outliers2, b=data_control_without_outliers2, equal_var=True)

print(result_stats2)
print("")

result2 = pg.ttest(data_reco_without_outliers2,
                  data_control_without_outliers2,
                  correction=True)
print(result2)
