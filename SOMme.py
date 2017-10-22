# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:03:16 2017

@author: buddy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset= pd.read_csv('Credit_Card_Applications.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som= MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
#random weights in som model
som.random_weights_init(X)
#traning som (euclidean distance)
som.train_random(data = X, num_iteration = 100)

#visualazing /plot som
from pylab import bone, pcolor, colorbar,plot, show
#init figure
bone()
#mean distance form between neurons function distance_map
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i, x in enumerate(X):
    w =  som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

#finding frauds
mappings = som.win_map(X)
#cordinates of winning node (1)
frauds = np.concatenate((mappings[(5,6)] , mappings[(5,7)]) , axis = 0)
frauds = sc.inverse_transform(frauds)
