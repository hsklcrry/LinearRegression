# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 17:20:34 2018

@author: igeh
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn


def corr(x,y):
    return ((x*y).mean() - x.mean()*y.mean()) / (((x*x).mean() - x.mean()**2) * ((y*y).mean() - y.mean()**2))**0.5

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

dtype = torch.float
input_data = torch.tensor([[94.2	,0.78	,50	,5],  \
[69.4	,0.56	,12	,4],  \
[97.9	,0.91	,16	,7],  \
[140.5	,0.85	,60	,2],  \
[135	,0.7	,55	,0],  \
[149.3	,0.77	,32	,0],  \
[169.8	,1.02	,42	,2],  \
[184.8	,1.12	,69	,3],  \
[47.9	,0.51	,8	,5],  \
[71.1	,0.69	,0	,6],  \
[77.7	,0.63	,22	,4],  \
[155	,0.9	,41	,2],  \
[157.2	,0.88	,39	,1],  \
[137.6	,0.84	,90	,2],  \
[106.5	,0.75	,18	,4]], dtype=dtype)

#for i in range(4):
#    input_data[:,i] /= ((input_data[:,i]*input_data[:,i]).mean())**0.5

input_dim = 3
output_dim = 1
x = input_data[:,1:4:1]
k = ((x*x).mean())**0.5
print(k)
x = x / k  # ((x*x).mean())**0.5
y = input_data[:,0:1]
y = y / ((y*y).mean())**0.5

model = LinearRegressionModel(input_dim,output_dim)


criterion = nn.MSELoss()  # Mean Squared Loss
l_rate = 0.01
optimiser = torch.optim.SGD(model.parameters(), lr = l_rate)
# Stochastic Gradient Descent

epochs = 5000
for epoch in range(epochs):
    inputs = Variable(x)
    labels = Variable(y)

    optimiser.zero_grad()

    outputs = model.forward(inputs)
    loss = criterion(outputs, labels)
    loss.backward()  # back props
    optimiser.step()  # update the parameters

predicted = model.forward(Variable(x))

try:
    xy = torch.tensor(list(zip(x,y)), dtype=dtype)
    x1 = xy[:,0]
    y1 = xy[:,1]
    
    plt.plot(x1.numpy(), y1.numpy(), 'go', label = 'исходные данные', alpha = .5)
    plt.plot(x1.numpy(), predicted.data.numpy(), label = 'прогноз', alpha = 0.5)
    plt.legend()
    plt.show()
except:
    print('')

a1, a0 = tuple(model.state_dict().values())
print('y = {a1}x + {a0}'.format(a1=a1, a0=a0.item()))
print('отклонение {}'.format(loss.item()))

r = ((x*y).mean() - x.mean()*y.mean()) / (((x*x).mean() - x.mean()**2) * ((y*y).mean() - y.mean()**2))**0.5
print('Коэф. корреляции: {}'.format(r))
    
