# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 09:52:41 2019

@author: OTIKO
"""

# Importing the libraires
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use('ggplot')
# Importing the dayaset
dataset = pd.read_csv( 'Position_Salaries.csv' )
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test)"""

# Every algorithm has a class where you import from
#Fitting Decision Tree Regression to the dataset
#randomstate is for random shaffling oir data

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

#Prediccting a new result
y_pred = regressor.predict([[6.5]])

#Visualising the Decision Tree Regression results (higher)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()