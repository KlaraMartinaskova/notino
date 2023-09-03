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
import datetime
import zipfile
import os
import csv


# # open CSV files
# with open('clients_final.csv', 'r') as file:
#     csv_clients = csv.reader(file)
# with open('orders_final.csv', 'r') as file:
#     csv_orders = csv.reader(file)

################################################################################################################
# Open csv file ans save as dataframe
df_clients = pd.read_csv('clients_final.csv',encoding="cp1250", sep=",", low_memory=False)
df_orders = pd.read_csv('orders_final.csv',encoding="cp1250", sep=",", low_memory=False)

################################################################################################################
# Analysis of abUser column
abUser_unique = df_clients['abUser'].unique() # 99 is not present here (just NaN)
txt = "Unique values in abUser: {}"
print(txt.format(abUser_unique))

################################################################################################################
# Devided into two groups by country
df_clients_ch = df_clients[df_clients['country']=="CH"]
df_clients_ne = df_clients[df_clients['country']=="NE"]

df_orders_ch = df_orders[df_orders['country']=="CH"]
df_orders_ne = df_orders[df_orders['country']=="NE"]

################################################################################################################
### Check the date
# •	Is the test running for the correct period?
def correct_period(df, start_date, end_date):
    df['date'] = pd.to_datetime(df['date'])
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    filtered_df['date'] = filtered_df['date'].dt.strftime('%Y-%m-%d')
    return filtered_df

start_date = datetime(2023, 5, 17)
end_date = datetime(2023, 6, 16)

df_clients_ch = correct_period(df_clients_ch, start_date, end_date) # data in correct period
df_clients_ne = correct_period(df_clients_ne, start_date, end_date) # data in correct period

################################################################################################################
### Select only unique clientID (for AB test we need to know assigment of each visitor to a group)
# Sort dataframe by date
df_clients_ch = df_clients_ch.sort_values(by=['date'])
df_clients_ne = df_clients_ne.sort_values(by=['date'])

# Drop duplicated client id and keep the first one
df_clients_ch = df_clients_ch.drop_duplicates(subset=['clientID'], keep='first')
df_clients_ne = df_clients_ne.drop_duplicates(subset=['clientID'], keep='first')

################################################################################################################
### Delete duplicates and NaN in orders

df_orders_ch = df_orders_ch.drop_duplicates(subset=["orderNumber"], keep='first') # drop duplicates
df_orders_ch= df_orders_ch.dropna(subset=["orderNumber"]) # drop NaN

df_orders_ne = df_orders_ne.drop_duplicates(subset=["orderNumber"], keep='first') # drop duplicates 
df_orders_ne= df_orders_ne.dropna(subset=["orderNumber"]) # drop NaN

################################################################################################################
""" •	Is the ratio of users in the reco group and users in the test group really 50:50? 
        Can you test it by an appropriate statistical test? 
        Do you prefer to test it on a daily basis, or to run one test for the whole period? 
        If you run multiple tests, do you need all of them to have positive results to verify the 50:50 distribution hypothesis? """


def binomial_test(df_group1,df_group2, expected_proportion, alt):
    observed_success = len(df_group1)
    total_items = len(df_group1) + len(df_group2)
    return stats.binomtest(observed_success, total_items, expected_proportion, alternative=alt)
   

def filter_day_and_abUser (df,day, abUser):
    return df[(df['date'] == unique_days[day]) & (df['abUser'] == abUser)]

def check_share_of_abUser_each_day (df,unique_days):
    for day in range(len(unique_days)):
        df_current_day_reco = filter_day_and_abUser(df,day,1)
        df_current_day_control = filter_day_and_abUser(df,day,2)
        share = 0.5
        binom = binomial_test(df_current_day_reco, df_current_day_control, share, "two-sided")

        if binom.pvalue < 0.05:
            print(f"\tDay {unique_days[day]} is significant")

### Test for each day
print("Test if the ratio of users in the reco group and users in the test group really 50:50\n")


# CH
unique_days = df_clients_ch['date'].unique()

print("Days with significant p-values for CH (if any): ")
check_share_of_abUser_each_day(df_clients_ch,unique_days)

# NE
unique_days = df_clients_ne['date'].unique()

print("Days with significant p-values for NE (if any): ")
check_share_of_abUser_each_day(df_clients_ne,unique_days)

print("---------------------------------")
### Test for whole period
print("Test ration for whole period: ")
share = 0.5

# CH
df_clients_ch_reco = df_clients_ch['abUser'][df_clients_ch['abUser']==1]
df_clients_ch_control = df_clients_ch['abUser'][df_clients_ch['abUser']==2]

binom_ch = binomial_test(df_clients_ch_reco , df_clients_ch_control, share, "two-sided")

print(f"Binomial test for CH: {binom_ch}")

if binom_ch.pvalue < 0.05:
    print("\tThe test is significant")
else:
    print("\tThe test is not significant")

# NE
df_clients_ne_reco = df_clients_ne['abUser'][df_clients_ne['abUser']==1]
df_clients_ne_control = df_clients_ne['abUser'][df_clients_ne['abUser']==2]

binom_ne = binomial_test(df_clients_ne_reco , df_clients_ne_control, share, "two-sided")

print(f"Binomial test for NE: {binom_ne}")

if binom_ne.pvalue < 0.05:
    print("\tThe test is significant")
else:
    print("\tThe test is not significant")

print("---------------------------------")
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
def binomial_test(df_group1,df_group2, expected_proportion, alt):
    observed_success = len(df_group1)
    total_items = len(df_group1) + len(df_group2)
    binom = stats.binomtest(observed_success, total_items, expected_proportion, alternative=alt)
    p_value = stats.binom_test(observed_success, total_items, expected_proportion, alternative=alt)
    return p_value, binom

# CH
assigned_group_ch = df_clients_ch['abUser'][(df_clients_ch['abUser']==1) | (df_clients_ch['abUser']==2)]
unassigned_group_ch = df_clients_ch[(df_clients_ch['abUser']).isna()]
p_value_ch, binom_ch = binomial_test(unassigned_group_ch, assigned_group_ch, 0.005, 'less')

print(f"Binomial test: {binom_ch}")
print(f"P-value: {p_value_ch}")

# NE
assigned_group_ne = df_clients_ne['abUser'][(df_clients_ne['abUser']==1) | (df_clients_ne['abUser']==2)]
unassigned_group_ne = df_clients_ne[(df_clients_ne['abUser']).isna()]
p_value_ne, binom_ne= binomial_test(unassigned_group_ne, assigned_group_ne, 0.005, 'less')

print(f"Binomial test for NE: {binom_ne}")
print(f"P-value: {p_value_ne}")
print("---------------------------------")
################################################################################################################
### •	What about the orders that are not in GA data? What is their share? How do you propose to handle them?
# CH
df_join_ch = df_clients_ch.merge(df_orders_ch, on="orderNumber", how="inner") # join tables (inner join)
share_orders_not_in_GA_ch = 1-(len(df_join_ch) / len(df_orders_ch)) # share of orders not in GA

print("Share of orders that are not in GA data for CH: {:.2f}%".format(share_orders_not_in_GA_ch *100))

# NE
df_join_ne = df_clients_ne.merge(df_orders_ne, on="orderNumber", how="inner") # join tables (inner join)
share_orders_not_in_GA_ne = 1-(len(df_join_ne) / len(df_orders_ne)) # share of orders not in GA

print("Share of orders that are not in GA data for CH: {:.2f}%".format(share_orders_not_in_GA_ne *100))
print("---------------------------------")
################################################################################################################
#  •	Does the “reco group” earn, on average, a greater revenue? Does it have larger orders? 
# Propose appropriate metrics and visualize them. Is there any other metric you may wish to evaluate?

# reco group =  abUser == 1
# control group = abUser == 2

def f_test(group1, group2):
    f = np.var(group1, ddof=1)/np.var(group2, ddof=1)
    nun = group1.size-1
    dun = group2.size-1
    p_value = 1-stats.f.cdf(f, nun, dun)
    return f, p_value

### Check the distribution of the data for quantity and revenue
df_join = df_join_ch
plt.scatter(df_join['quantity'][df_join['abUser']==1], df_join['revenue'][df_join['abUser']==1], alpha=0.5, marker='o')
plt.scatter(df_join['quantity'][df_join['abUser']==2], df_join['revenue'][df_join['abUser']==2], alpha=0.5, marker = "x")
plt.xlabel('Quantity')
plt.ylabel('Revenue')
plt.legend(['reco group', 'control group'])
plt.title('Distruibution of revenue and quantity for reco and control group for country CH')
plt.show()

df_join = df_join_ne
plt.scatter(df_join['quantity'][df_join['abUser']==1], df_join['revenue'][df_join['abUser']==1], alpha=0.5, marker='o')
plt.scatter(df_join['quantity'][df_join['abUser']==2], df_join['revenue'][df_join['abUser']==2], alpha=0.5, marker = "x")
plt.xlabel('Quantity')
plt.ylabel('Revenue')
plt.legend(['reco group', 'control group'])
plt.title('Distruibution of revenue and quantity for reco and control for country NE')
plt.show()

# Drop outliers from NE dataframe
df_join_without_outliers = df_join_ne.drop(reco_max)

# Plot of distribution without outliers
plt.scatter(df_join_without_outliers['quantity'][df_join_without_outliers['abUser']==1], df_join_without_outliers['revenue'][df_join_without_outliers['abUser']==1], alpha=0.5, marker='o')
plt.scatter(df_join_without_outliers['quantity'][df_join_without_outliers['abUser']==2], df_join_without_outliers['revenue'][df_join_without_outliers['abUser']==2], alpha=0.5, marker = "x")
plt.xlabel('Quantity')
plt.ylabel('Revenue')
plt.legend(['reco group', 'control group'])
plt.title('Distruibution of revenue and quantity for reco and control group without outliers for NE')
plt.show()

df_join_ne = df_join_without_outliers # refined dataframe

###############
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


