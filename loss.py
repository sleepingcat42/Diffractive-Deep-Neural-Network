# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:41:18 2023

@author: sleepingcat
github: https://github.com/sleepingcat42
e-mail: sleepingcat@aliyun.com
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