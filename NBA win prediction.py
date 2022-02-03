# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 20:46:39 2021

@author: bakel
"""

import numpy as np
import pandas as pd

#Loading dataset and data pre-process
data = pd.read_csv('games_details.csv', sep = ',', header = 0)
data.isna().sum()
data.drop(['START_POSITION', 'COMMENT'], axis = 1, inplace = True)
data.dropna(axis = 0, inplace = True)

players = pd.read_csv('players.csv', sep = ',', header = 0)

df = data.drop(['TEAM_ABBREVIATION', 'TEAM_CITY', 'PLAYER_ID', 'PLAYER_NAME', 
                'MIN'], axis = 1).groupby(['GAME_ID', 'TEAM_ID']).sum()
plus_minus = data.drop(['TEAM_ABBREVIATION', 'TEAM_CITY', 'PLAYER_ID', 'PLAYER_NAME', 'MIN'], axis = 1).groupby(
    ['GAME_ID', 'TEAM_ID'])['PLUS_MINUS'].sum()/data.groupby(['GAME_ID', 'TEAM_ID']).size()
plus_minus = pd.DataFrame(plus_minus, columns = ['PLUS_MINUS'])
df['PLUS_MINUS'] = plus_minus['PLUS_MINUS']
df.reset_index(level = 0, inplace = True)


teams = data[['TEAM_ID', 'TEAM_ABBREVIATION']]
teams = teams[~teams.TEAM_ID.duplicated(keep = 'first')]

df = df.merge(teams, how = 'left', on = 'TEAM_ID')
df.drop(['TEAM_ID'], axis = 1, inplace = True)

df['FG_PCT'] = df['FGM'].values / df['FGA'].values
df['FG3_PCT'] = df['FG3M'].values / df['FG3A'].values
df['FT_PCT'] = df['FTM'].values / df['FTA'].values

df1 = df.groupby('GAME_ID').first()
df2 = df.groupby('GAME_ID').tail(1)
df2.set_index(['GAME_ID'], inplace = True)
df2 = df2.add_suffix('_y')

games = pd.concat([df1, df2], axis = 1)
games.drop(['TEAM_ABBREVIATION'], axis = 1, inplace = True)

#Corellation matrix
import seaborn as sns
import matplotlib.pyplot as plt
#Corr Matrix gia ola ta features
sns.set(style = "white")
cor_matrix = games.loc[:, 'FGM': 'PLUS_MINUS_y'].corr()
mask = np.triu(np.ones_like(cor_matrix, dtype = np.bool))
plt.figure(figsize = (15, 12))
cmap = sns.diverging_palette(220, 10, as_cmap = True)
sns.heatmap(cor_matrix, mask = mask, cmap = cmap, center = 0,
            square = True, linewidths = .5, cbar_kws = {"shrink": .5})

#Selecting features for final model
X_df1 = pd.concat([df1[['FG_PCT', 'FG3_PCT', 'FT_PCT','AST', 'PLUS_MINUS']],
                 df2[['PLUS_MINUS_y', 'DREB_y', 'BLK_y']]], axis = 1)
X_df1.reset_index(drop = True, inplace = True)
X_df2 = pd.concat([df2[['FG_PCT_y', 'FG3_PCT_y', 'FT_PCT_y','AST_y', 'PLUS_MINUS_y']],
                 df1[['PLUS_MINUS', 'DREB', 'BLK']]], axis = 1)
X_df2.reset_index(drop = True, inplace = True)
X_df2.rename(columns = {'FG_PCT_y' : 'FG_PCT', 'FG3_PCT_y' :'FG3_PCT', 'FT_PCT_y' : 'FT_PCT','AST_y' :'AST',
                        'PLUS_MINUS_y' : 'PLUS_MINUS', 'PLUS_MINUS' : 'PLUS_MINUS_y',
                        'DREB' : 'DREB_y', 'BLK' : 'BLK_y'}, inplace = True)
df2.rename(columns = {'PTS_y' : 'PTS'}, inplace = True)
X = pd.concat([X_df1, X_df2], axis = 0)
y = pd.concat([df1['PTS'],df2['PTS']], axis = 0)
y = y.reset_index(drop = True)
y = pd.DataFrame(y, columns = ['PTS'])


#Data scaling
from sklearn.preprocessing import  StandardScaler
sc1 = StandardScaler()
sc2 = StandardScaler()
X = sc1.fit_transform(X)
y = sc2.fit_transform(y)

import sklearn
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
# Split data using train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)



#RBf SVM model
clf = SVR(kernel="rbf", C=1, gamma = 0.1, epsilon = 0)

rbf_pred = clf.fit(x_test,y_test.ravel())
y_predrbf = clf.predict(x_test)

acc1 = clf.score(x_test, y_test.ravel())

from sklearn.metrics import mean_squared_error
import math
y_predrbf = pd.DataFrame(y_predrbf, columns = ['PTS_x'])
y1_inverse = sc2.inverse_transform(y_predrbf)
y1_test_inverse = sc2.inverse_transform(y_test)

#Model accuracy
MSE = mean_squared_error(y1_test_inverse, y1_inverse)
RMSE = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(RMSE)

#Predicting whole dataset as unknown
train_pred = clf.predict(x_train)
train_pred = pd.DataFrame(train_pred, columns = ['PTS'])
train = sc2.inverse_transform(train_pred)
train_y = sc2.inverse_transform(y_train)
train_acc = clf.score(x_train, y_train)

#Model accuracy
MSE2 = mean_squared_error(train, train_y)
RMSE2 = math.sqrt(MSE2)
print("Root Mean Square Error:\n")
print(RMSE2)

#Linear SVM model
clf2 = SVR(kernel="linear", C=1)

rbf_pred2 = clf2.fit(x_test,y_test)
y_predrbf2 = clf2.predict(x_test)

acc2 = clf.score(x_test, y_test)

y_predrbf2 = pd.DataFrame(y_predrbf2, columns = ['PTS_y'])
y2_inverse = sc2.inverse_transform(y_predrbf2)
y2_test_inverse = sc2.inverse_transform(y_test)

#Model accuracy
MSE2 = mean_squared_error(y2_test_inverse, y2_inverse)
RMSE2 = math.sqrt(MSE2)
print("Root Mean Square Error:\n")
print(RMSE2)

#Cross-Validation
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

train_sizes, train_scores, valid_scores = learning_curve(
    SVR(kernel="rbf", C=1, gamma = 0.1, epsilon = 0), X, y, scoring = 'neg_mean_squared_error',
    train_sizes= [1, 5000, 10000, 25000, 37704], cv=5, n_jobs = -1)

train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -valid_scores.mean(axis = 1)

plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a linear regression model', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,1)

#Tuning the parameters
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1],'kernel': ['rbf', 'linear']}
grid = GridSearchCV(sklearn.svm.SVR(),param_grid,refit=True,verbose=2)
grid.fit(x_train,y_train) 
Best = grid.best_estimator_
print(grid.best_estimator_)


#Function for predicting an NBA game score
stats = pd.read_csv('NBA sim.csv', sep = ';', header = 0)

def predict(df):
    print(stats['Team'])
    print('Pick the first team: ')
    team1 = stats.loc[stats['Team'] == str(input(''))].reset_index(drop = True).add_suffix('_x')
    print('Pick the second team: ')
    team2 = stats.loc[stats['Team'] == str(input(''))].add_suffix('_y').reset_index(drop = True)    
    X_team1 = pd.concat([team1[['FG_PCT_x', 'FG3_PCT_x', 'FT_PCT_x','AST_x', 'PLUS_MINUS_x']],
                     team2[['PLUS_MINUS_y', 'DREB_y', 'BLK_y']]], axis = 1)
    X_team1.rename(columns = {'FG_PCT_x' : 'FG_PCT', 'FG3_PCT_x' : 'FG3_PCT','FT_PCT_x' : 'FT_PCT', 
                              'AST_x' : 'AST', 'PLUS_MINUS_x' : 'PLUS_MINUS'}, inplace = True)
    print(X_team1)
    X_team2 = pd.concat([team2[['FG_PCT_y', 'FG3_PCT_y', 'FT_PCT_y','AST_y', 'PLUS_MINUS_y']],
                     team1[['PLUS_MINUS_x', 'DREB_x', 'BLK_x']]], axis = 1)
    X_team2.rename(columns = {'FG_PCT_y' : 'FG_PCT', 'FG3_PCT_y' : 'FG3_PCT', 'FT_PCT_y': 'FT_PCT','AST_y' : 'AST', 
                              'PLUS_MINUS_y' : 'PLUS_MINUS', 'PLUS_MINUS_x' : 'PLUS_MINUS_y',
                              'DREB_x' : 'DREB_y', 'BLK_x' : 'BLK_y'}, inplace = True)
    print(X_team2)
    X_team1 = sc1.transform(X_team1)
    print(X_team1)
    X_team2 = sc1.transform(X_team2)
    print(X_team2)
    team1_predict = clf.predict(X_team1)
    team2_predict = clf.predict(X_team2)
    team1_predict = pd.DataFrame(team1_predict, columns = ['PTS_1'])
    team1_predict = sc2.inverse_transform(team1_predict)
    team1_predict = int(team1_predict)
    print(team1_predict)
    team2_predict = pd.DataFrame(team2_predict, columns = ['PTS_2'])
    team2_predict = sc2.inverse_transform(team2_predict)
    team2_predict = int(team2_predict)
    print(team2_predict)
    if team1_predict > team2_predict:
        print('Team 1 wins')
    elif team2_predict > team1_predict:
        print('Team 2 wins')
    
predict(stats)




























