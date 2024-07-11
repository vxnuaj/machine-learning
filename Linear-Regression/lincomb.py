import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Pred
def predict(x, m, b):
    pred = np.dot(x, m) + b
    return pred

# Loss
def MSE(y, pred):
    loss = np.mean((y - pred) ** 2) 
    return loss

# Gradient Descent
def gradient_descent(x, y, m, b, epochs, alpha):
    for i in range(epochs):
        pred = predict(x, m, b)

        m_gradient = -2 * np.dot(x.T, (y - pred))
        b_gradient = -2 * (y-pred)

        m = m - alpha * m_gradient
        b = b - alpha * b_gradient
    
    print(f"Slope: {m}")
    print(f"Bias: {b}")
    return m, b

if __name__ == "__main__":
    data = pd.read_csv("./data/random.csv")
    data = np.array(data)    
    scalar = StandardScaler()
    x = scalar.fit_transform(data[:,0:2])
    y = scalar.fit_transform(data[:,2].reshape(-1,1))

    n = len(x)

    m = np.random.rand(2, 1)
    b = np.random.rand()

    epochs = 500
    alpha = .0001

    m, b = gradient_descent(x, y, m, b, epochs, alpha)