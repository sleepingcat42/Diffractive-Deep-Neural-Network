# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:05:41 2022

@author: Chen Chunyuan
"""
# import torch
from torch import nn
from OpticalLayers import DiffLayer,  Diffraction

class Onn(nn.Module):
    def __init__(self, M, L, lambda0, z):
        super(Onn, self).__init__()
        self.layer0 = Diffraction(M, L, lambda0, z[0])
        self.DiffLayer1 = DiffLayer(M, L, lambda0, z[1])
        self.DiffLayer2 = DiffLayer(M, L, lambda0, z[2])
        self.DiffLayer3 = DiffLayer(M, L, lambda0, z[3])

    def forward(self, u1):        
        u = self.layer0(u1)
        u = self.DiffLayer1(u)
        u = self.DiffLayer2(u)
        u = self.DiffLayer3(u)

        return u
    
