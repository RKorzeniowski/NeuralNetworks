# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 12:32:27 2017

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

#Convert ratings into binary ratings 1(like) 0(not liked) -1(not rated)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

#Creating architeture of the neural network
    #def Class

    #(object that will be created,numver of vis nodes,numver of hidden nodes)

        #a is of hidden node(batch_size ,bias)        randn has normal distribution mean 0 variance 1

        #b is biad of the vis node

    #sampling/activation hidden nodes x is vis node
    #this returns probability that neuron in hidden layer based on input is active

class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)


#Training the RBM
nb_epoch = 10

#for number of echo
for epoch in range(1, nb_epoch +1):
    #error between orginal function and prediction    
    train_loss = 0
    #counter
    s = 0.
    #for batch size
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        # ,_ return only 1st argument of function that retunrs 2 arguments
        ph0,_ = rbm.sample_h(v0)
        #for k steps of constactive divergence
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>0] - vk[v0>0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
        
        
#Testing RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    v_target = test_set[id_user:id_user+1]
    if len(v_target[v_target>0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)    
        test_loss += torch.mean(torch.abs(v_target[v_target>0] - v[v_target>0]))
        s += 1.
print('test loss: '+str(test_loss/s))
            
        
        