# NEURAL NINE'S IMPLEMENTATION - FOR COMPARISON PURPOSES TO LINREGSELF.PY AND LINRSCIKIT.PY

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./Data/random1.csv')

x = data.iloc[:202].x
y = data[:202].y

def loss_function(m, b, points, x, y):
    total_error = 0

    for i in range(len(points)):
        x = points.iloc[i].x
        y = points.iloc[i].y
        total_error += (y - (m * x + b)) ** 2
    total_error / float(len(points))

def gradient_descent(m_now, b_now, points, alpha, x, y):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].x
        y = points.iloc[i].y

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now ))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - alpha * m_gradient
    b = b_now - alpha * b_gradient
    return m, b

m = 0
b = 0
alpha = .0001
epochs = 1000

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, data, alpha, x, y)

print(m, b)

plt.scatter(x, y, color = "black")
plt.plot(x, (m * x + b), color = "red")
plt.show()
