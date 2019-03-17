# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:09:41 2018

@author: IV_Ernst
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

Z = torch.tensor([ \
[80, 11.34, 83.1, 85.2], \
[20, 11.34, 60.6, 62.5], \
[80, 9.75, 71.8, 73.9], \
[20, 9.75, 83.7, 81.9],])

Y = Z[:,2:4]
Y_m = Y.mean(1)

D = ((Y[:,0] - Y_m)**2) + ((Y[:,1] - Y_m)**2)

X = torch.zeros((4,2))
for i in range(2):
    print(Z[:,i].mean(), (Z[:,i] - Z[:,i].mean()).max())
    X[:,i] = (Z[:,i] - Z[:,i].mean()) / (Z[:,i] - Z[:,i].mean()).max()

b = torch.zeros(3)
b[0] = Y_m.mean()
for i in range(2):
    b[i + 1] = (Y_m * X[:,i]).mean()

c = torch.zeros(3)
c[0] = b[0]
for i in range(1, 3):
    c[i] = b[i] / (Z[:,i - 1] - Z[:,i - 1].mean()).max()
    c[0] = c[0] - Z[:,i - 1].mean()/(Z[:,i - 1] - Z[:,i - 1].mean()).max()


f = 4 * (2 - 1)
s_v = D.mean()
s_b = s_v / (4 * 2) ** 0.5
print(c / s_b)
# 2.776 для 0.05


