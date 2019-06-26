################################################
# RNN_model.ipynb
# 
# Author: Aude Forcione-Lambert
# 
# Date: june 4th 2019
# 
# Description: PyTorch custom single-layered RNN class for training on toy datasets
################################################

from pathlib import Path
import requests
from itertools import chain
import pickle
import gzip
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math

# General purpose single-layered RNN
class Toy_RNN(nn.Module):
    
    # Class instanciator
    # n_inputs: number of inputs per time step
    # n_neurons: number of neurons in the hidden layer
    # n_outputs: number of outputs
    # loss_func: loss function. Must be part of the torch.nn.functional library
    # opt_func: optimizing function. Must be part of the torch.optim library
    # lr: learning rate for the optimizing function
    def __init__(self, n_inputs, n_neurons, n_outputs, loss_func, opt_func, lr):
        super().__init__()
        
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        
        self.rnn = nn.RNN(n_inputs, n_neurons, bias=True)
        self.lin = nn.Linear(n_neurons, n_outputs, bias=False)
        
        self.opt = opt_func(chain(self.rnn.parameters(),self.lin.parameters()), lr=lr)
        self.loss_func = loss_func
        
    # Defines a forward pass
    # x: input tensor of shape batch_size x n_inputs x n_inputs
    # return: network output at the last time step of shape batch_size x n_outputs
    #         all_hidden: hidden states of the RNN layer computed for every input in the given batch (including the final output of the RNN layer)
    def forward(self, x):
        # transforms x to dimensions: n_inputs X batch_size X n_inputs
        x = x.permute(1, 0, 2)
        
        batch_size = x.size(1)
        self.hidden = torch.zeros(1, batch_size, self.n_neurons)

        all_hidden, self.hidden = self.rnn(x, self.hidden)
        # hidden are the hidden states on the last element of each input, aka the final output of the RNN layer
        # all_hidden contains the hidden states for every element of each input, self.hidden is the last element of all_hidden
        # index: #28 seq_len, 2000 examples in each batch (batch_size), 200 neurons (the last 200 are in tensor, rest is list)
        
        out = self.lin(self.hidden)
        
        return out.view(-1, self.n_outputs), all_hidden  #batch_size X n_output

    # Trains the network to reach a target behavior
    # epochs: Number of training epochs
    # train_dl: pyTorch DataLoader object containing the training dataset in (x,y) pairs
    # return: vector containing the total loss per training epoch, list
    def fit(self, epochs, train_dl,valid_dl):
        
        parameters = list(self.get_params())
        hidden_states = list()
        loss_array=np.zeros(epochs)

        for epoch in range(epochs):
            epoch_parameters= list()
            epoch_states = list()
            

            for xb,yb in train_dl:

                self.opt.zero_grad()
                prediction, batch_states = self.forward(xb)
                loss = self.loss_func(prediction, yb)
                loss_array[epoch] += loss
                loss.backward()
                self.opt.step()

                epoch_parameters.append(self.get_params())
                epoch_states.append(batch_states)
            
            parameters.append(epoch_parameters)
            hidden_states.append(epoch_states)
            
            print('Epoch: ' , epoch)
            print('Accuracy: ' , self.accuracy(valid_dl).item())
        
        return loss_array, parameters, hidden_states
    
    # Returns the network's parameters
    # parameters is a list indexed by: # epoch, #batch, # layer (0: input, 1: hidden, 2: output)
    # each layer is a numpy array of dimensions: 
    # input: (# neurons, input_size), hidden: (# neurons, # neurons), output: (n_output, # neurons)
    # see torch.nn.RNN ~variables for more details
    # in general 1st index defines where the connexion is going (output node) and the 2nd index
    # defines where the connexion is comming from (input node)
    def get_params(self):
        input_weights = self.rnn.weight_ih_l0.data.numpy()
        hidden_weights = self.rnn.weight_hh_l0.data.numpy()
        output_weights = self.lin.weight.data.numpy()
        return input_weights, hidden_weights, output_weights

# Single-layered RNN for classification tasks
class RNN_classifier(Toy_RNN):
    
    # Network accuracy testing with human-readable output
    # valid_dl: pyTorch DataLoader object containing the testing dataset in (x,y) pairs
    # return: proportion of times when the network accurately predicted the class of the input (where the predicted class is the class that matched with highest confidence)
    def accuracy(self, valid_dl):
        x_valid, y_valid = next(iter(valid_dl))
        preds = torch.argmax(self.forward(x_valid)[0], dim=1)
        return (preds == y_valid).float().mean()
    
# Single-layered RNN for target-value matching task
class RNN_target_value(Toy_RNN):
    
    # Network accuracy testing with human-readable output
    # valid_dl: pyTorch DataLoader object containing the testing dataset in (x,y) pairs
    # threshold: tolerance threshold for difference between actual and expected output
    # return: proportion of times when the network accurately matched the expected output (norm of difference between predicted and expected vectors smaller than threshold)
    def accuracy(self, valid_dl, threshold):
        x_valid, y_valid = next(iter(valid_dl))
        preds = self.forward(x_valid)[0]
        return torch.le(torch.norm(preds-y_valid,dim=1),threshold).float().mean()
