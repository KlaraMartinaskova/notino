# Notino Test Assigment
# Author: Klara Martinaskova
# Task: Evaluate an AB test of the recommendation algorithm

#!/usr/bin/env python3.9
# coding=utf-8

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
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


if reco_mean > control_mean:
    print("The reco group earns, on average, a greater revenue.")
else:
    print("The reco group does not earn, on average, a greater revenue.")

print("The average revenue of reco group is {:.2f} and the average revenue of control group is {:.2f}".format(reco_mean, control_mean))
    