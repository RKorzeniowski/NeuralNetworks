# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 19:39:05 2017

@author: buddy
"""
import matplotlib.pyplot as plt
        
import numpy as np
import pandas as pd

traning_set = pd.read_csv('Google_Stock_Price_Train.csv')
#iloc to get the indexes u want to extract (expects matrix not a vektor so instead of [:,1] i get [:,1:2]) but 2 is exluded
traning_set = traning_set.iloc[:,1:2]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#transformiaton of scaling data for min=0 and max=1
traning_set = scaler.fit_transform(traning_set)

#getting the imputs and the outputs (all 1258 but we dont know the output for the last one)
X_train = traning_set[0:1257]
#shifted by 1 "day" compared to X_train 
y_train = traning_set[1:1258]

#reshapeing 3d array (3rd is time step)
X_train = np.reshape(X_train, (1257, 1, 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#init the RNN
regressor = Sequential()

#adding the input layer and the LSTM later
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

#adding output layer (units=number of neurons)
regressor.add(Dense(units = 1))

#compile the RNN (optimizer 'rnsprop','adam')
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#fitting RNN to traning set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)

#getting real stock price
test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2]

#getting the preficted stock price of 2017
inputs = real_stock_price
inputs = scaler.transform(inputs)
inputs = np.reshape(inputs, (20, 1, 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

#visualasing the results
plt.plot(real_stock_price, color = 'red', label = 'real stock price')
plt.plot(predicted_stock_price, color = 'blue', label = 'predicted stock price')
plt.title('Google stock price predictions')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()



#p2

#getting the real stock price for 2012 2016
real_stock_price_train = pd.read_csv('Google_Stock_Price_Train.csv')
real_stock_price_train = real_stock_price_train.iloc[:,1:2]

#getting the predicted stock price of 2012 2016
predicted_stock_price_train = regressor.predict(X_train)
predicted_stock_price_train = scaler.inverse_transform(predicted_stock_price_train)

#vis the results
plt.plot(real_stock_price_train, color = 'red', label = 'real stock price')
plt.plot(predicted_stock_price_train, color = 'blue', label = 'predicted stock price')
plt.title('Google stock price predictions')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()

#p2 end


# Evaluating the RNN
import math
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price)) 


