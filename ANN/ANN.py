# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 21:03:50 2017

@author: buddy
"""

# Part 1 Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#dummy var trap on country
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]



# Splitting the dataset into the Training set and Test set
# change form "cross_validation" to "model_selection" becouse of patch in sypder
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Part2 ANN

#importing Keras
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialasing the ANN as sequence of layers
classifier = Sequential()

#adding first input layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = "relu", input_dim = 11))

#add 2nd layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = "relu"))

#add output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = "sigmoid"))

#compiling the ANN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])

#fitting ANN to the traning set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)



# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#homework pred
new_predction = classifier.predict(sc.transform(np.array([[0,0, 600, 1,40,3,60000,2,1,1,50000]])))
new_predction = (new_predction > 0.5)


X_train = sc.fit_transform(X_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Part4 evaluation improving and tuning the ANN
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
#dropout prevents overfitting in NN by turing off random neurons in each iteration and making predictions with different configurations
from keras.layers import Dropout


def build_classifier():
    #Initialasing the ANN as sequence of layers
    classifier = Sequential()

    #adding first input layer with dropout
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = "relu", input_dim = 11))
    classifier.add(Dropout(p = 0.1))
    #add 2nd layer with dropout
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = "relu"))
    classifier.add(Dropout(p = 0.1))
    #add output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = "sigmoid"))

    #compiling the ANN
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])
    
    return classifier
    
#trained with k-fold cross validation
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
#mean of 10 accurasies form 10k fold validation
mean = accuracies.mean()
#.std to get the variance
variance = accuracies.std()


import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
#dropout prevents overfitting in NN by turing off random neurons in each iteration and making predictions with different configurations
from keras.layers import Dropout


def build_classifier1(optimizer,nb_layers1,nb_layers2):
    #Initialasing the ANN as sequence of layers
    classifier = Sequential()

    #adding first input layer
    classifier.add(Dense(output_dim = nb_layers1, init = 'uniform', activation = "relu", input_dim = 11))

    #add 2nd layer
    classifier.add(Dense(output_dim = nb_layers2, init = 'uniform', activation = "relu"))

    #add output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = "sigmoid"))

    #compiling the ANN
    classifier.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ['accuracy'])
    
    return classifier
    
classifier = KerasClassifier(build_fn = build_classifier1)
#creat a dictionary
parameters = {'batch_size': [25],
              'nb_epoch': [250],
#'adam'
              'optimizer': ['rmsprop'],
              'nb_layers1': [10],
              'nb_layers2': [10]}
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = 4)
                           
grid_search = grid_search.fit(X_train, y_train)
best_parameters_1 = grid_search.best_params_ 
best_accuracy_1 = grid_search.best_score_





