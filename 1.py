
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

input_dim = 1
output_dim = 1

X = dict()
Y = dict()

dtype = torch.float
X[0] = torch.tensor([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=dtype)
Y[0] = torch.tensor([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.56], dtype=dtype)

X[1] = torch.tensor([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=dtype)
Y[1] = torch.tensor([9.14, 8.14, 8.74, 8.77, 9.26, 8.1, 6.13, 3.1, 9.13, 7.26, 4.74], dtype=dtype)

X[2] = torch.tensor([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=dtype)
Y[2] = torch.tensor([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73], dtype=dtype)

X[3] = torch.tensor([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], dtype=dtype)
Y[3] = torch.tensor([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.5, 5.56, 7.91, 6.89], dtype=dtype)

# X[4] = torch.tensor([0,1,2,3,4,5,6,7,8,9,10], dtype=dtype)
# Y[4] = torch.tensor([10,9,8,7,6,5,4,3,2,1,0], dtype=dtype)


for (x, y) in zip(X.values(), Y.values()):

    model = LinearRegressionModel(input_dim,output_dim)

    criterion = nn.MSELoss()  # Mean Squared Loss
    l_rate = 0.007
    optimiser = torch.optim.SGD(model.parameters(), lr = l_rate) 
    # Stochastic Gradient Descent

    epochs = 2000
    for epoch in range(epochs):
        epoch +=1

        inputs = Variable(x.reshape(11,1))
        labels = Variable(y.reshape(11,1))

        optimiser.zero_grad()

        outputs = model.forward(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()  # back props
        optimiser.step()  # update the parameters

    predicted = model.forward(Variable(x.reshape(11,1))).data.numpy()

    plt.plot(x.numpy(), y.numpy(), 'go', label = 'исходные данные', alpha = .5)
    plt.plot(x.numpy(), predicted, label = 'прогноз', alpha = 0.5)
    plt.legend()
    plt.show()

    a1, a0 = tuple(model.state_dict().values())
    print('y = {a1}x + {a0}'.format(a1=a1.item(), a0=a0.item()))
    print('отклонение {}'.format(loss.item()))

    r = ((x*y).mean() - x.mean()*y.mean()) / (((x*x).mean() - x.mean()**2) * ((y*y).mean() - y.mean()**2))**0.5
    print('Коэф. корреляции: {}'.format(r))

