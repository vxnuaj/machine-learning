import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def init_params():
    rng = np.random.default_rng(seed = 1)
    w = rng.normal(size = (1, 2))
    b = np.zeros((1, 1))
    return w, b

def forward(x, w, b):
    z = np.dot(w, x) + b
    return z 

def mse(y, z):
    loss = np.sum(np.square(y - z)) / y.size
    return loss

def backwards(y, x, z):
    dz = (-2 * (y - z))
    dw = np.dot(dz, x.T) / y.size
    db = np.sum(dz) / y.size
    return dw, db
    
def update(w, b, dw, db, alpha):
    w -= alpha * dw
    b -= alpha * db
    return w, b
    
def gradient_descent(x, y, w, b, alpha, epochs):
    for epoch in range(epochs):
        z = forward(x, w, b)
        
        loss = mse(y, z)
        
        dw, db = backwards(y, x, z)
        w, b = update(w, b , dw, db, alpha)
      
        if epoch % 20 == 0: 
            print(f"Epoch: {epoch}") 
            print(f"Loss: {loss}\n")   

    print(f"linear regression from scratch using sgd in numpy")
    print(f"W: {w}")
    print(f"B: {b}") 
    print(f"Final MSE: {loss}")
    return w, b

if __name__ == "__main__":
    data = pd.read_csv('data/quad.csv')
    data = np.array(data)

    X_train = data[:, :2]
    Y_train = data[:, 2].reshape(-1, 1)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train).T
    Y_train = ss.fit_transform(Y_train).reshape(1, -1)
    
    w, b = init_params()
    alpha = .001
    epochs = 500000

    gradient_descent(X_train, Y_train, w, b, alpha, epochs)