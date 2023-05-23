# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:41:18 2023

@author: sleepingcat
github: https://github.com/sleepingcat42
e-mail: sleepingcat@aliyun.com
"""

import torch
from torch import nn

from torch.fft import fft2, fftshift, ifft2, ifftshift


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
PI = torch.pi

# M: sampling rate
# L: Length of the network
# lambda0: wavelength

class DiffLayer(nn.Module):
    def __init__(self, M, L, lambda0, z):
        super(DiffLayer, self).__init__()
        # self.params=nn.Parameter(torch.rand(1, M, M))*2*torch.pi
        self.params=nn.Parameter(torch.zeros(1, M, M))
        self.H = self.get_kernel(M, L, lambda0, z)
        
    def forward(self, u1):
        u1 = u1*torch.exp(1j*self.params)
        U1 = fft2(fftshift(u1))
        U2 = U1*self.H
        return ifftshift(ifft2(U2))
    
    def get_kernel(self, M, L, lambda0, z):
        
        dx = L/M
        k = 2 * PI / lambda0
        fx = torch.linspace(-1/(2*dx), 1/(2*dx)-1/L, M)   
        FX, FY = torch.meshgrid(fx, fx, indexing='xy')
        
        A=1 - ((lambda0 *FX)**2 + (lambda0 *FY)**2)
        A = A+0j
        H = torch.exp(1j * k * z * torch.sqrt(A))
        H = fftshift(H)
        H = H.to('cuda:0')
        return H.unsqueeze(0)    
    
    def phase_init(self, phase):
        self.params = phase

class Diffraction(nn.Module):
    def __init__(self, M, L, lambda0, z):
        super(Diffraction, self).__init__()
        self.H = self.get_kernel(M, L, lambda0, z)
        self.X, self.Y = self.get_gridXY(M, L)
        self.L = L
        
    def forward(self, u1):

        U1 = fft2(fftshift(u1))
        U2 = U1*self.H
        return ifftshift(ifft2(U2))
    
    def get_kernel(self, M, L, lambda0, z):
        
        dx = L/M
        k = 2 * PI / lambda0
        fx = torch.linspace(-1/(2*dx), 1/(2*dx)-1/L, M)   
        FX, FY = torch.meshgrid(fx, fx, indexing='ij')
        
        A=1 - ((lambda0 *FX)**2 + (lambda0 *FY)**2)
        A = A+0j
        H = torch.exp(1j * k * z * torch.sqrt(A))
        H = fftshift(H)
        H = H.to('cuda:0')
        return H.unsqueeze(0)    
    
    def get_gridXY(self, M, L):
        dx = L/M
        x = torch.linspace(-L/2, L/2-dx, M).to(device)   
        X, Y = torch.meshgrid(x,x, indexing='ij')
        return X, Y
