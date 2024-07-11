# PYTORCH IMPLEMENATION FOR SKILLSETS + COMPARISON TO LINREGSELF.PY

import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression(nn.Module):

    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)
    


data = pd.read_csv("./Data/random1.csv")
data = np.array(data)

x = data[:, 0]
y = data[:, 1]

x = torch.tensor(x, dtype = torch.float32)
y = torch.tensor(y, dtype = torch.float32)

x = x.reshape(-1,1)
y = y.reshape(-1,1)


input_size = 1
output_size = 1
model = LinearRegression(input_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = .00001)

epochs = 2000

for i in range(epochs):
    outputs = model(x)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 100 == 0:
        print(f'Epoch [{i+1}/{epochs}], Loss: {loss.item():.4f}')

print("Prediction:", model(torch.tensor([54.88135039273247])))
print("Weight:", model.linear.weight.data)
print("Bias:", model.linear.bias.data)


plt.scatter(x, y, color='blue', label='Original data')


with torch.no_grad():
    predicted = model(x)
plt.plot(x, predicted, color='red', label='Fitted line')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()