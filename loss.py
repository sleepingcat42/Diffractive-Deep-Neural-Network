# -*- coding: utf-8 -*-
"""
Created on Fri May 27 17:20:14 2022

@author: Chen
"""
import torch


def npcc_loss(X, Y):
    X_mean = torch.mean(X, dim=(2,3), keepdim=True)
    Y_mean = torch.mean(Y,dim=(2,3), keepdim=True) 
    a = torch.sum((X - X_mean)*(Y - Y_mean), dim=(2,3), keepdim=True)
    b = torch.sqrt(torch.sum((X - X_mean)**2, dim=(2,3), keepdim=True)
                   *torch.sum((Y - Y_mean)**2, dim=(2,3), keepdim=True))
    npcc = torch.mean(-a/b)
    return npcc