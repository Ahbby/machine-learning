# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:37:37 2019

@author: OTIKO
"""

# Simple Linear Regression

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
style.use('ggplot')

#Importingg the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Splitting the database into the Training set and Test set...
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Feature Scaling

#Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test SET results


#Visualising the Training set results
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.ylabel('Salary')
plt.show()