# SELF IMPLEMENTATION FROM FIRST PRINCIPLES THINKING.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Prediction

def predict(m, b, x):
    pred = m * x + b
    return pred

# Loss | MSE

def MSE(y, pred ):
    loss = np.mean((y - pred)**2)
    return loss

# Gradient Descent

def gradient_descent(m, b, alpha, epochs, x, y, loss):
    for epoch in range(epochs):
        pred = predict(m, b, x)

        m_gradient = -2 * np.mean(x * (y - (pred)))
        b_gradient = -2 * np.mean(y - pred)

        m -= alpha * m_gradient
        b -= alpha * b_gradient

        print(f"Epoch: {epoch}")
    
        loss = MSE(y, pred)
        print(f"Param m: {m}")
        print(f"Param b: {b}")
        print(f"Loss: {loss}")


    return m, b


if __name__ == "__main__":

    data = pd.read_csv("./Data/random1.csv")
    data = np.array(data)

    x = data[:, 0] # AGE
    y = data[:, 1]

    m = 0.1
    b = 3

    alpha = .00001
    epochs = 20000

    loss = 0

    m, b = gradient_descent(m, b, alpha, epochs, x, y, loss)