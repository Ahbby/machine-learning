# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:00:16 2019

@author: OTIKO
"""
#Importing the libraries
import numpy as py
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib import style 
style.use('ggplot')

# Importing the dataset
# we use a list when we're subseting more than two items.

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_test)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Visualising the Training set results
from matplotlib.colors import ListedColormap
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:,0.max]))
                     np.arange()
