# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 09:49:11 2019

@author: OTIKO
"""

#data preprocessing template

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the datasetre
# X refers to features
#y refers to the targets
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#in machine learning, 
#splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

#Taking care of missing data
#the variable 'imputer' is imported to find our missing values
#the fit method checks our function in the dataset
#by default

from sklearn.preprocessing import Imputer])

imputer = Imputer(missing_values = 'NaN' , strategy = 'mean' , axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#LABEL ENCODER - It will help us to replace our categorical data into numeric data, by using the 'dummy varaiable 
# for your y' values that's your target(purchase) you don't need dummy variable.

#Encoding categorical data
#Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# coding the dependent variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Pandas is a library for doing data analysis in python.

# Feature Scaling
# SCALING IS TRIMMING DOWN FEATURE VALUES TO HAVE A MEAN OF 0 AND STANDARD DEVIATION OF 1(THAT'S TRIMMING IT TO BECOME SMALL)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
