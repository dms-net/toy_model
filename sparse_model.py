################################################
# sparse_model.py
# 
# Author: Aude Forcione-Lambert
# 
# Date: june 16 2019
# 
# Description: 
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

class sparse_model(nn.Module):
    
    def __init__(self, n_inputs, n_neurons, n_outputs, loss_func, opt_func, lr):
        super().__init__()
        
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        
        self.w = torch.randn(n_neurons, n_inputs).to_sparse().requires_grad_(True)
        self.wout = torch.randn(n_outputs, n_neurons).requires_grad_(True)
        
        self.opt = opt_func(chain(self.w.parameters(),self.wout.parameters()), lr=lr)
        self.loss_func = loss_func
        
    def forward(self, x):
        s = torch.sparse.mm(w,x.t())
        return torch.mm(wout,s)
    
    def fit(self, epochs, train_dl):
        loss_array=np.zeros(epochs)
        for epoch in range(epochs):
            for xb,yb in train_dl:
                loss = self.loss_func(self.forward(xb), yb)
                loss_array[epoch] += loss
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
            print(epoch)
        return loss_array
    
    def accuracy(self, valid_dl):
        x_valid, y_valid = next(iter(valid_dl))
        preds = torch.argmax(self.forward(x_valid), dim=0)
        return (preds == y_valid).float().mean()