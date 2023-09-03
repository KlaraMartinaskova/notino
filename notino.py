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

# Devided into two groups by country
df_clients_ch = df_clients[df_clients['country']=="CH"]
df_clients_ne = df_clients[df_clients['country']=="NE"]

################################################################################################################
####Check the date
# •	Is the test running for the correct period?
def correct_period(df, start_date, end_date):
    df['date'] = pd.to_datetime(df['date'])
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    filtered_df['date'] = filtered_df['date'].dt.strftime('%Y-%m-%d')
    return filtered_df

start_date = datetime(2023, 5, 17)
end_date = datetime(2023, 6, 16)

df_clients_ch = correct_period(df_clients_ch, start_date, end_date)
df_clients_ne = correct_period(df_clients_ne, start_date, end_date)
################################################################################################################
# Analysis of abUser column
abUser_unique = df_clients['abUser'].unique() # 99 is not present here (just NaN)
txt = "Unique values in abUser: {}"
print(txt.format(abUser_unique))
################################################################################################################
#### •	What about the users with an unassigned group? Bambino thinks the test is fine if their share is below 0.5%. 
# First method
def unassigned_share(df_clients):
    reco_count = df_clients['abUser'][df_clients['abUser']==1].count()
    control_count = df_clients['abUser'][df_clients['abUser']==2].count()
    other_count = (df_clients['abUser']).isna().sum()
    percent = other_count/((reco_count + control_count + other_count))*100

    if percent < 0.5:
        print("The test is fine, share is below 0.5%.")
        print("Percent of users with an unassigned group: {:.2f}%".format(percent))
    else:
        print("The test is not fine, share is {:.2f}%".format(percent))

    return percent

share_ch = unassigned_share(df_clients_ch)
share_ne = unassigned_share(df_clients_ne)

# Second method with binomial test
def binomial_test(df_group1,df_group2, expected_proportion, alt ):
    observed_success = len(df_group1)
    total_items = len(df_group1) + len(df_group2)
    binom = stats.binomtest(observed_success, total_items, expected_proportion, alternative=alt)
    p_value = stats.binom_test(observed_success, total_items, expected_proportion, alternative=alt)
    print(f"Binomial test: {binom}")
    print(f"P-value: {p_value}")
    return p_value

# CH
assigned_group_ch = df_clients_ch['abUser'][(df_clients_ch['abUser']==1) | (df_clients_ch['abUser']==2)]
unassigned_group_ch = df_clients_ch[(df_clients_ch['abUser']).isna()]
p_value_ch = binomial_test(unassigned_group_ch, assigned_group_ch, 0.005, 'less')
print(p_value_ch)

# NE
assigned_group_ne = df_clients_ne['abUser'][(df_clients_ne['abUser']==1) | (df_clients_ne['abUser']==2)]
unassigned_group_ne = df_clients_ne[(df_clients_ne['abUser']).isna()]
p_value_ne = binomial_test(unassigned_group_ne, assigned_group_ne, 0.005, 'less')
print(p_value_ne)


################################################################################################################
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
plt.title('Distruibution of revenue and quantity for reco and control group')
plt.show()

### detected outliner, which does not change the mean quantity result

reco_max = df_join['revenue'][df_join['abUser']==1].idxmax()

df_join_without_outliers = df_join.drop(reco_max)

# Plot of distribution without outliers
plt.scatter(df_join_without_outliers['quantity'][df_join_without_outliers['abUser']==1], df_join_without_outliers['revenue'][df_join_without_outliers['abUser']==1], alpha=0.5, marker='o')
plt.scatter(df_join_without_outliers['quantity'][df_join_without_outliers['abUser']==2], df_join_without_outliers['revenue'][df_join_without_outliers['abUser']==2], alpha=0.5, marker = "x")
plt.xlabel('Quantity')
plt.ylabel('Revenue')
plt.legend(['reco group', 'control group'])
plt.title('Distruibution of revenue and quantity for reco and control group without outliers')
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

# Boxplot for revenue
sns.boxplot(data=df_join_without_outliers, x='abUser', y='revenue')
plt.ylabel("Revenue")
plt.title("Box Plot for Reco and Control Group - Revenue")
plt.show()

# Boxplot for quantity
sns.boxplot(data=df_join_without_outliers, x='abUser', y='quantity')
plt.ylabel("Number of items in the order")
plt.title("Box Plot for Reco and Control Group - Quantity")
plt.show()

# Histogram for quantity
plt.hist(x = df_join['quantity'][df_join['abUser']==1], bins = 100,  alpha = 0.5, label = 'reco group')
plt.hist(x = df_join['quantity'][df_join['abUser']==2], bins = 100,  alpha = 0.5, label = 'control group')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.legend(['reco group', 'control group'])
plt.title('Histogram of quantity for reco and control group')
plt.show()

# Visualization of average revenue by quantity
grouped_data = df_join_without_outliers.groupby(['quantity', 'abUser'])['revenue'].mean().unstack()

ax = grouped_data.plot(kind='bar', alpha=0.7)
ax.set_title('Average revenue by quantity')
ax.set_ylabel('Average revenue')
ax.set_xlabel('Quantity')
ax.legend(['reco group', 'control group'])
plt.show()


# Visualization of average revenue by quantity for most common quantity
filtered_df = df_join_without_outliers[df_join_without_outliers['quantity'].between(1, 10)]
grouped_data_filtered = filtered_df.groupby(['quantity', 'abUser'])['revenue'].mean().unstack()

ax = grouped_data_filtered.plot(kind='bar', alpha=0.7)
ax.set_title('Average revenue by quantity')
ax.set_ylabel('Average revenue')
ax.set_xlabel('Quantity')
ax.legend(['reco group', 'control group'])
plt.show()

### Diference between country

# Number of orders by country
grouped_data_by_country = df_join_without_outliers.groupby(['country_y'])['quantity'].count()
ax = grouped_data_by_country.plot(kind='bar', alpha=0.7)
ax.set_title("Number of orders by country")
ax.set_ylabel('Number of makings orders')
ax.set_xlabel('country')
plt.show()

# Frequency of revenue by country
plt.hist(x = df_join_without_outliers['revenue'][df_join['country_y']=="CH"], bins = 100,  alpha = 0.5, label = 'ch')
plt.hist(x = df_join_without_outliers['revenue'][df_join['country_y']=="NE"], bins = 100,  alpha = 0.5, label = 'ne')
plt.xlabel('Revenue')
plt.ylabel('Frequency')
plt.legend()
plt.title('Histogram of revenue by country')
plt.show()

# Frequency of quantity by country
plt.hist(x = df_join_without_outliers['quantity'][df_join['country_y']=="CH"], bins = 100,  alpha = 0.5, label = 'ch')
plt.hist(x = df_join_without_outliers['quantity'][df_join['country_y']=="NE"], bins = 100,  alpha = 0.5, label = 'ne')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.legend()
plt.title('Histogram of quantity by country')
plt.show()

# Boxplot for quantity
sns.boxplot(data=df_join_without_outliers, x='country_y', y='quantity')
plt.ylabel("Number of items in the order")
plt.title("Box Plot for country - Quantity")
plt.show()


