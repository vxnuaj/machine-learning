from logreg import load_model, forward, log_loss
import pandas as pd
import numpy as np

def make_pred(w, b, x, y):
    a = forward(x, w, b)
    for i in range(len(x)):

        pred = a[i]
        true_val = y[i]

        pred = np.max(pred)

        loss = log_loss(pred, true_val)     
        print(f"Prediction: {pred}")
        print(f"True Value: {true_val}")
        print(f"Loss: {loss}")

if __name__ == "__main__":
    w, b = load_model('models/linreg.pkl')

    data = pd.read_csv("./Data/randomtest.csv")
    data = np.array(data)
    
    x_test = data[:, 0:2]
    y_test = data[:, 2].reshape(-1,1)

    make_pred(w, b, x_test, y_test)

