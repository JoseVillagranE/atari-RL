#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 02:04:10 2021

@author: josev
"""


import torch
import torch.nn as nn


def conv2d_size_out(size, kernels_size, strides, paddings, dilations):
    for kernel_size, stride, padding, dilation in zip(kernels_size, strides, paddings, dilations):
        size = (size + 2*padding - dilation*(kernel_size - 1) - 1)//stride + 1
    return size

class DQN(nn.Module):
    
    def __init__(self, action_space, n_channel):
        super().__init__()
        self.action_space = action_space
        
        self.conv = nn.Sequential(nn.Conv2d(n_channel, 32, kernel_size=1), 
                                  nn.Conv2d(32, 64, kernel_size=8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                  nn.ReLU())
        
        outp_size = conv2d_size_out(84, [8, 4, 3], [4, 2, 1], [0, 0, 0], [1, 1, 1])
        outp_size= 64*outp_size**2
        
        self.linear = nn.Sequential(nn.Linear(outp_size, 512), 
                                    nn.ReLU(),
                                    nn.Linear(512, action_space),
                                    nn.Softmax(-1))
        
        
    def forward(self, x):
        action = self.linear(torch.flatten(self.conv(x), start_dim=1))
        return action
        
    
if __name__=="__main__":
    
    input = torch.rand((1, 3, 84, 84))
    model = DQN(18, 3)
    outp = model(input)