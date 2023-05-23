# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:41:18 2023

@author: sleepingcat
github: https://github.com/sleepingcat42
e-mail: sleepingcat@aliyun.com
"""

import torch


def rect(X,w):
    Y = torch.zeros(X.shape)
    Y[torch.abs(X)<w]=1
    return Y.cuda()

L = 80
M = 256
dx = L/M
x1 = torch.linspace(-L/2, L/2-dx, M)
X1, Y1 = torch.meshgrid(x1, x1, indexing='ij')
w = 2
yb = 7
xb = 7

cens = [[-xb, yb],[0, yb], [xb, yb],
        [-xb/2-xb,0],[-xb/2,0],[xb/2,0],[xb/2+xb,0],
        [-xb, -yb],[0, -yb], [xb, -yb]]
cens2 = [] 
w2 = int( w*M/L+1)
# labels = label_generator()
for x, y in cens:
    cens2.append([int(M/2+x/L*M),int(M/2+y/L*M)]) 

def label_generator():

    labels = torch.zeros(10,M,M)
    for k in range(10):
        labels[k] = rect(X1-cens[k][0],w)*rect(Y1-cens[k][1],w)
    return labels.unsqueeze(1)


def eval_accuracy(target, label):
        
    label_hat = torch.zeros(target.size(0), 10).cuda()
    # print(label_hat[:,1].size())
    for k in range(10):
        x, y = cens2[k]
        # label_hat[:,k] = target[:, :, y-w2:y+w2, x-w2:x+w2].sum(dim=(2,3)).squeeze(1)
        label_hat[:,k] = target[:, :, x-w2:x+w2, y-w2:y+w2].sum(dim=(2,3)).squeeze(1)          
    # print(label_hat.size())
    label_hat = label_hat.argmax(1)
    acc = (label_hat == label.cuda()).sum()/label.size(0)
    
    return acc.data, label_hat

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test = torch.tensor([0,1,1,3,4,3,6,7,8,9])
    a = label_generator()
    target = a[test]
    acc = eval_accuracy(target, test)
    a = torch.sum(a, axis=0)
    plt.imshow(a[0],cmap='gray')
    plt.axis('equal')