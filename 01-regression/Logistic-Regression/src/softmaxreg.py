import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import preprocessing as skpp

def init_params():
    w = np.random.randn(3, 4)
    b = np.random.randn(3, 1)
    return w, b

def softmax(z):
    eps = 1e-10
    z = z.astype(float)
    return np.exp(z + eps) / np.sum(np.exp(z+ eps), axis=0, keepdims=True)

def forward(x, w, b):
    z = np.dot(w, x) + b # 3, 1
    a = softmax(z) # 3, 1
    return a

def one_hot(y):
    lencode = skpp.LabelEncoder()
    ly = lencode.fit_transform(y)

    one_hot_y = np.zeros((np.max(ly) + 1, 150)) #3, 150
    one_hot_y[ly, np.arange(ly.size)] = 1
    return one_hot_y

def loss(one_hot_y, a):
    l = - np.sum(one_hot_y * np.log(a)) / 150
    return l

def backward(x, one_hot_y, a):
    dz = a - one_hot_y
    dw = np.dot(dz, x.T) / 150
    db = np.sum(dz, axis=1, keepdims=True) / 150
    return dw, db

def update(dw, db, w, b, alpha):
    w = w - alpha * dw
    b = b - alpha * db
    return w, b

def gradient_descent(x, y, w, b, alpha, epochs):
    one_hot_y = one_hot(y)
    for epoch in range(epochs):
        a = forward(x, w, b)
        
        l = loss(one_hot_y, a)

        dw, db = backward(x, one_hot_y, a)
        w, b = update(dw, db, w, b, alpha)

        if epoch % 500 == 0:
            print(f"epoch: {epoch}")
            print(f"loss: {l}")

    return w, b

def model(x, y, alpha, epochs):
    w, b = init_params()
    w, b = gradient_descent(x, y, w, b, alpha, epochs)
    return w, b

if __name__=="__main__":
    data = pd.read_csv('Data/iris.csv')
    data = np.array(data)

    X_train = data[:, 0:4].T
    Y_train = data[:, 4]
    print(X_train.dtype)

    model(X_train, Y_train, .01, 10000)



