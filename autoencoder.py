# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:50:44 2017

@author: buddy
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#Importing the dataset
directory_movies ='/home/buddy/Documents/neural networks tutorial/A-Z neural networks tutaorial/Deep Learning A-Z/Volume 2 - Unsupervised Deep Learning/Part 5 - Boltzmann Machines (BM)/Boltzmann_Machines/ml-1m/movies.dat'
movies = pd.read_csv(directory_movies , sep = '::', header = None, engine = 'python', encoding = 'latin-1')
directory_users ='/home/buddy/Documents/neural networks tutorial/A-Z neural networks tutaorial/Deep Learning A-Z/Volume 2 - Unsupervised Deep Learning/Part 5 - Boltzmann Machines (BM)/Boltzmann_Machines/ml-1m/users.dat'
users = pd.read_csv(directory_users , sep = '::', header = None, engine = 'python', encoding = 'latin-1')
directory_ratings ='/home/buddy/Documents/neural networks tutorial/A-Z neural networks tutaorial/Deep Learning A-Z/Volume 2 - Unsupervised Deep Learning/Part 5 - Boltzmann Machines (BM)/Boltzmann_Machines/ml-1m/ratings.dat'
ratings = pd.read_csv(directory_ratings , sep = '::', header = None, engine = 'python', encoding = 'latin-1')

#Preparing the training set and the test set
directory_training_set = '/home/buddy/Documents/neural networks tutorial/A-Z neural networks tutaorial/Deep Learning A-Z/Volume 2 - Unsupervised Deep Learning/Part 5 - Boltzmann Machines (BM)/Boltzmann_Machines/ml-100k/u1.base'
training_set = pd.read_csv(directory_training_set, delimiter = '\t')
#Convert dataset to array
training_set = np.array(training_set, dtype = 'int')

#Preparing the test set and the test set
directory_test_set = '/home/buddy/Documents/neural networks tutorial/A-Z neural networks tutaorial/Deep Learning A-Z/Volume 2 - Unsupervised Deep Learning/Part 5 - Boltzmann Machines (BM)/Boltzmann_Machines/ml-100k/u1.test'
test_set = pd.read_csv(directory_test_set, delimiter = '\t')
#Convert dataset to array
test_set = np.array(test_set, dtype = 'int')

#Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

#Converting the data into an array with users in lies and movies in colums rating in cells
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        #[fuction][condition] 
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
    
training_set = convert(training_set)
test_set = convert(test_set)    

#Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#Creatubg the architecture of the neural network
#stacked auto encoders
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        #(,size of hidden layer)
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        #activation function
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

#Traning the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        #check if user rated at least one movie
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

#Testset

#Getting the number of users and movies
test_loss = 0
s = 0.

for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    #check if user rated at least one movie
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))

