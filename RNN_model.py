################################################
# RNN_model.ipynb
# 
# Author: Aude Forcione-Lambert
# 
# Date: june 4th 2019
# 
# Description: 
################################################

from pathlib import Path
import requests
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

class Toy_RNN(nn.Module):
    
    def __init__(self, n_steps, n_inputs, n_neurons, n_outputs):
        super().__init__()
        
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        
        self.rnn = nn.RNN(n_inputs, n_neurons, bias=True)
        self.lin = nn.Linear(n_neurons, n_outputs, bias=False)
        
    def forward(self, x):
        # transforms x to dimensions: n_steps X batch_size X n_inputs
        x = x.permute(1, 0, 2)
        
        batch_size = x.size(1)
        self.hidden = torch.zeros(1, batch_size, self.n_neurons)
        
        rnn_out, self.hidden = self.rnn(x, self.hidden)
        
        out = self.lin(self.hidden)
        
        return out.view(-1, self.n_outputs) # batch_size X n_output
    
    def fit(self, epochs, train_dl):
        loss_array=np.zeros(epochs)
        for epoch in range(epochs):
            for xb,yb in train_dl:
                loss = self.loss_func(self.forward(xb), yb)
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
            loss_array[epoch] = loss
            print(epoch)
        return loss_array
    
    @staticmethod
    def get_model(n_steps, n_inputs, n_neurons, n_outputs, loss_func, opt_func, lr):
        model = Toy_RNN(n_steps, n_inputs, n_neurons, n_outputs)
        model.opt = opt_func(model.parameters(), lr=lr)
        model.loss_func = loss_func
        return model
    

class RNN_classifier(Toy_RNN):
    
    def accuracy(self, valid_dl):
        x_valid, y_valid = next(iter(valid_dl))
        preds = torch.argmax(self.forward(x_valid), dim=1)
        return (preds == y_valid).float().mean()
    
    @staticmethod
    def get_model(n_steps, n_inputs, n_neurons, n_outputs, loss_func, opt_func, lr):
        model = RNN_classifier(n_steps, n_inputs, n_neurons, n_outputs)
        model.opt = opt_func(model.parameters(), lr=lr)
        model.loss_func = loss_func
        return model
