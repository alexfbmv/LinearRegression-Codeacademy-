# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 17:42:06 2022

@author: Alex
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# load and investigate the data here:
    df = pd.read_csv('tennis_stats.csv')



## perform single feature linear regressions here:
    
# Model_1: Predicting Ranking from number of Aces
# store columns into x and y variables
x0 = df['Aces']
y0 = df['Ranking']

# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=60)

# create LR model
mlr = LinearRegression()
mlr.fit(x_train, y_train)

# print scores
print(mlr.score(x_train, y_train))
print(mlr.score(x_test, y_test))

# predict y_values
y_predicted = mlr.predict(x_train)



# Model_2 (predicting Winnings from Break opps.)
x0 = df['BreakPointsOpportunities']
y0 = df['Winnings']

x = x0.values.reshape(-1,1)
y = y0.values.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=60)

mlr = LinearRegression()
mlr.fit(x_train, y_train)

print(mlr.score(x_train, y_train))
print(mlr.score(x_test, y_test))

y_predicted = mlr.predict(x_train)




## perform two-feature linear regressions here:

# Model 1 (predicting yearly earnings from RGP and SGW)
df = pd.read_csv('tennis_stats.csv')
df.info()

x = df[['ReturnGamesPlayed','ServiceGamesWon']]
y = df['Winnings']

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, test_size = 0.2, random_state = 60)

mlr = LinearRegression()
mlr.fit(x_train,y_train)

y_predicted = mlr.predict(x_train)

print(mlr.score(x_train,y_train))
print(mlr.score(x_test,y_test))
#0.8342773292711096
#0.8266077533085391

plt.scatter(df['ReturnGamesPlayed'],df['Winnings'], alpha=0.4, s=2)
plt.scatter(df['ServiceGamesWon'],df['Winnings'], alpha=0.4, s=2)



# double model 2 (predicting yearly earnings from FS and Wins)
x = df[['FirstServe','Wins']]
y = df['Winnings']

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, test_size = 0.2, random_state = 60)

mlr = LinearRegression()
mlr.fit(x_train,y_train)

print(mlr.score(x_train,y_train))
print(mlr.score(x_test,y_test))
#0.83427732927110963
#0.8266077533085391

plt.scatter(df['FirstServe'],df['Winnings'], alpha=0.4, s=2)
plt.scatter(df['Wins'],df['Winnings'], alpha=0.4, s=2)



# double model 3 (predicting yearly earnings from Losses and Wins)
x = df[['Losses','Wins']]
y = df['Winnings']

y = y.values.reshape (-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, test_size = 0.2, random_state = 60)

mlr = LinearRegression()
mlr.fit(x_train,y_train)

y_predicted = mlr.predict(x_train)

print(mlr.score(x_train,y_train))
print(mlr.score(x_test,y_test))

#0.8590372334012266
#0.8327730449655714

plt.scatter(df['Losses'],df['Winnings'], alpha=0.4, s=2)
plt.scatter(df['Wins'],df['Winnings'], alpha=0.4, s=2)




## perform multiple feature linear regressions here:
    
# Multiple linear regressin: model_1
df = pd.read_csv('tennis_stats.csv')
df.info()

x = df[['FirstServe','FirstServeReturnPointsWon','SecondServeReturnPointsWon',
        'Aces','DoubleFaults','ReturnGamesPlayed','ReturnPointsWon',
        'ServiceGamesPlayed', 'ServiceGamesWon','TotalPointsWon',
        'Wins','Losses']]

y = df['Winnings']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=60)

mlr = LinearRegression()
mlr.fit(x_train, y_train)

print(mlr.score(x_train, y_train))
print(mlr.score(x_test, y_test))
# 0.8717540437545561
# 0.848097126604829
print(mlr.coef_)


    
# MLR model_2
x = df[['FirstServePointsWon','SecondServePointsWon',
        'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
        'BreakPointsSaved','DoubleFaults', 'ReturnGamesPlayed',
        'ReturnGamesWon','TotalServicePointsWon',]]

y = df['Winnings']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=60)

mlr = LinearRegression()
mlr.fit(x_train, y_train)


print(mlr.score(x_train, y_train))
print(mlr.score(x_test, y_test))
#0.8425238419077565
#0.8352515056802541

print(mlr.coef_)