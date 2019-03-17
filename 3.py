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
#input_data = torch.tensor(
#[
#[91.8	,1.1	,90,	-1 ], \
#[75.8	,1.1	,90,	3  ], \
#[88.55	,1.1	,25,	-1 ], \
#[72.55	,1.1	,25,	3  ], \
#[90.6	,0.7	,90,	-1 ], \
#[74.6	,0.7	,90,	3  ], \
#[87.35	,0.7	,25,	-1 ], \
#[71.35	,0.7	,25,	3  ]
#], dtype=dtype)
#    
input_data = torch.tensor(
[
[91.8	,1.1	,90,	-1 ], \
[75.8	,1.1	,90,	3  ], \
[88.55	,1.1	,25,	-1 ], \
[72.55	,1.1	,25,	3  ], \
[90.6	,0.7	,90,	-1 ], \
[74.6	,0.7	,90,	3  ], \
[87.35	,0.7	,25,	-1 ], \
[71.35	,0.7	,25,	3  ]
], dtype=dtype)

A = torch.zeros(8,4) # (8, 4)
A[:,0] = input_data[:,0]

for i in range(0,4):
    max1 = input_data[:,i].max().item()
    min1 = input_data[:,i].min().item()
    center = (max1 + min1) / 2
    A[:,i] = 2 * (input_data[:,i] - center) / (max1 - min1)

b = torch.zeros(4, 4)

b[0,0] = A[:, 0].mean()
b[0,1] = ( A[:, 1] * A[:, 0]).mean()
b[0,2] = ( A[:, 2] * A[:, 0]).mean()
b[0,3] = ( A[:, 3] * A[:, 0]).mean()

c = torch.tensor([3, 0.05, -4])

#b[1,0] = A[:4, 0].mean()
#b[1,1] = ( A[:4, 2] * A[:4, 3] * A[:4, 0]).mean()
#b[1,2] = ( A[:4, 2] * A[:4, 0]).mean()
#b[1,3] = ( A[:4, 3] * A[:4, 0]).mean()
#
#b[2,0] = A[:, 0].mean()
#b[2,1] = ( A[:, 1] * A[:, 0]).mean()
#b[2,2] = ( A[:, 1] * A[:, 3] * A[:, 0]).mean()
#b[2,3] = ( A[:, 3] * A[:, 0]).mean()
#


#
#for i in range(4):
#    input_data[:,i] /= ((input_data[:,i]*input_data[:,i]).mean())**0.5
#
#input_dim = 3
#output_dim = 1
#x = input_data[:,1:4:1]
#k = ((x*x).mean())**0.5
#print(k)
##x = x / k  # ((x*x).mean())**0.5
#y = input_data[:,0:1]
##y = y / ((y*y).mean())**0.5
#
#model = LinearRegressionModel(input_dim,output_dim)
#
#
#criterion = nn.MSELoss()  # Mean Squared Loss
#l_rate = 0.05
#optimiser = torch.optim.SGD(model.parameters(), lr = l_rate)
## Stochastic Gradient Descent
#
#epochs = 500
#for epoch in range(epochs):
#    inputs = Variable(x)
#    labels = Variable(y)
#
#    optimiser.zero_grad()
#
#    outputs = model.forward(inputs)
#    loss = criterion(outputs, labels)
#    loss.backward()  # back props
#    optimiser.step()  # update the parameters
#
#predicted = model.forward(Variable(x))
#
#try:
#    xy = torch.tensor(list(zip(x,y)), dtype=dtype)
#    x1 = xy[:,0]
#    y1 = xy[:,1]
#    
#    plt.plot(x1.numpy(), y1.numpy(), 'go', label = 'исходные данные', alpha = .5)
#    plt.plot(x1.numpy(), predicted.data.numpy(), label = 'прогноз', alpha = 0.5)
#    plt.legend()
#    plt.show()
#except:
#    print('')
#
#a1, a0 = tuple(model.state_dict().values())
#print('y = {a1}x + {a0}'.format(a1=a1, a0=a0.item()))
#print('отклонение {}'.format(loss.item()))
#
#r = ((x*y).mean() - x.mean()*y.mean()) / (((x*x).mean() - x.mean()**2) * ((y*y).mean() - y.mean()**2))**0.5
#print('Коэф. корреляции: {}'.format(r))
#    
