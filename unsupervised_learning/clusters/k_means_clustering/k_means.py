# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:41:22 2019

@author: OTIKO
"""

#K-Means Clustering

#Importing the libraries
import numpy as py
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use('ggplot')

#Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X_ dataset.iloc[:, [3,4]].values
#y = dataset.iloc[:, 3].values

#Splitting the dataset into the Training set and Test set
