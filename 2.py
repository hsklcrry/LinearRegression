# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 20:32:52 2018

@author: igeh
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = 4
output_dim = 1

dtype = torch.float
  # Length	 Weight	Len^3	Ln(Length)	Ln(Weight)
input_data = torch.tensor([[58,	28,	195112,	4.060443,	3.33220451], \
[61,	44,	226981,	4.1108739,	3.78418963],\
[63,	33,	250047,	4.1431347,	3.49650756],\
[68,	39,	314432,	4.2195077,	3.66356165],\
[69,	36,	328509,	4.2341065,	3.58351894],\
[72,	38,	373248,	4.2766661,	3.63758616],\
[72,	61,	373248,	4.2766661,	4.11087386],\
[74,	54,	405224,	4.3040651,	3.98898405],\
[74,	51,	405224,	4.3040651,	3.93182563],\
[76,	42,	438976,	4.3307333,	3.73766962],\
[78,	57,	474552,	4.3567088,	4.04305127],\
[82,	80,	551368,	4.4067192,	4.38202664],\
[85,	84,	614125,	4.4426513,	4.4308168 ],\
[86,	83,	636056,	4.4543473,	4.41884061],\
[86,	80,	636056,	4.4543473,	4.38202664],\
[86,	90,	636056,	4.4543473,	4.49980967],\
[88,	70,	681472,	4.4773368,	4.24849524],\
[89,	84,	704969,	4.4886364,	4.4308168 ],\
[90,	106,	729000,	4.4998097,	4.66343909],\
[90,	102,	729000,	4.4998097,	4.62497281],\
[94,	110,	830584,	4.5432948,	4.70048037],\
[94,	130,	830584,	4.5432948,	4.86753445],\
[114,	197,	1481544,	4.7361984,	5.28320373],\
[128,	366,	2097152,	4.8520303,	5.90263333],\
[147,	640,	3176523,	4.9904326,	6.46146818]], dtype=dtype)


#X = dict()
#Y = dict()
#Z = dict()
#ds = dict()
#
#
#X[0] = torch.tensor([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=dtype)
#Y[0] = torch.tensor([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.56], dtype=dtype)
    
#for (x, y) in zip(X.values(), Y.values()):
x = input_data[:,1:5]
y = input_data[:,0:1]

model = LinearRegressionModel(input_dim,output_dim)

criterion = nn.MSELoss()  # Mean Squared Loss
l_rate = 0.007
optimiser = torch.optim.SGD(model.parameters(), lr = l_rate) 
# Stochastic Gradient Descent

epochs = 200
for epoch in range(epochs):
    inputs = Variable(x)
    labels = Variable(y)

    optimiser.zero_grad()

    outputs = model.forward(inputs)
    loss = criterion(outputs, labels)
    loss.backward()  # back props
    optimiser.step()  # update the parameters

predicted = model.forward(Variable(x))

plt.plot(x.numpy(), y.numpy(), 'go', label = 'исходные данные', alpha = .5)
plt.plot(x.numpy(), predicted.data.numpy(), label = 'прогноз', alpha = 0.5)
plt.legend()
plt.show()

a1, a0 = tuple(model.state_dict().values())
print('y = {a1}x + {a0}'.format(a1=a1, a0=a0.item()))
print('отклонение {}'.format(loss.item()))

r = ((x*y).mean() - x.mean()*y.mean()) / (((x*x).mean() - x.mean()**2) * ((y*y).mean() - y.mean()**2))**0.5
print('Коэф. корреляции: {}'.format(r))
    
