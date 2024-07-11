import numpy as np
import pandas as pd
import sys
import heartdisease
from heartdisease import forward, load_model, log_loss


def model_test(x, y, filename):
    try:
        w, b = load_model(filename)
        print("Model loaded!")

    except FileNotFoundError:
        sys.exit("Model doesn't exist! Train one first!")
    
    a = forward(x, w, b) # DIMS (1, 54)

    loss = log_loss(a, y)     

    a = np.round(a, decimals=0)

    for i in range(len(a[0])):
        print(f"{i} | True Value: {y[i]}")
        print(f"{i} | Prediction: {a[0, i]}")
        print("\n")

    print(f"Test loss: {loss}")

    return


if __name__ == "__main__":
    
    data = pd.read_csv('./Data/heart.csv')
    data = np.array(data)

    X_test_norm = data[249:, :13].T # DIMS: (13, 54)
    Y_test = data[249:, 13].reshape(54, -1) #DIMS: (54, 1)

    #X_test_scaled = (X_test_norm - np.min(X_test_norm, keepdims=True, axis = 0)) / (np.max(X_test_norm, keepdims = True, axis = 0) - np.min(X_test_norm, keepdims = True, axis = 0))

    filename = './models/HeartLogReg.pkl'

    model_test(X_test_norm, Y_test, filename)

