'''
File paths are fidel to the working directory of "Logistic-Regression".

Of the test dataset "heart.csv", the former 250 samples will be training and latter 53 samples will be the testing set.
'''

'''
SIDE NOTES: N/A
'''

import numpy as np
import pandas as pd
import pickle


def save_model(w, b, filename):
    with open(filename, 'wb') as f:
        pickle.dump((w, b), f)

def load_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def init_params():
    w = np.random.rand(1, 13)
    b = np.random.rand(1, 1)
    return w, b

def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a

def forward(x, w, b):
    z = np.dot(w, x) + b
    a = sigmoid(z)
    return a #DIMS: 1, 249

def log_loss(a, y): #Log Loss
    eps = 1e-10
    loss = - np.mean(y * np.log(a + eps) + (1 - y) * np.log(1-a + eps))
    return loss

def back_prop(w, b, x, y, a, alpha):
    dw = np.mean((a - y) * x)
    db = np.mean((a - y))
    w = w - alpha * dw
    b = b - alpha * db
    return w, b 

def gradient_descent(x, y, w, b, alpha, epochs):

    for epoch in range(epochs):
        a = forward(x, w, b)
        loss = log_loss(a, y)
        w, b = back_prop(w, b, x, y, a, alpha)

        print(f"Epoch: {epoch}")
        print(f"Loss: {loss}")

    print(a.shape)
    return w, b

def model_train(x, y, alpha, epochs, filename):

    try:
        w, b = load_model(filename)
        print(f"Loading model from {filename}!")
    except FileNotFoundError:
        print("Model not found, initializing new params!")
        w, b = init_params()
    
    w, b = gradient_descent(x, y, w, b, alpha, epochs)
    save_model(w, b, filename)
    print("Finished! Model updated!")

    return

if __name__ == "__main__":
    data = pd.read_csv('./Data/heart.csv')
    data = np.array(data)

    filename = './models/HeartLogReg.pkl'

    X_train = data[:249, :13].T # DIMS: (249, 13)
    Y_train = data[:249, 13].reshape(249, -1).T # DIMS: (249, 1)

    '''
    Normalizing data between 0 and 1: norm_data = (x - x_min) / (x_max - x_min)
    '''
    #X_train = (X_train - np.min(X_train, axis = 0, keepdims = True)) / (np.max(X_train, axis = 0, keepdims = True) - np.min(X_train, axis = 0, keepdims = True))


    model_train(X_train, Y_train, .001, 300000, filename)


''''
Reflection:

- Loading / saving models isn't seamless for me might need to learn how / practice how
- Gonna need to practice pre processing and normalizing data. Knowing when to and how to.
- Need to brush on up linear algebra. I shouldn't be asking questionso on how element wise subtraction works lol. Should gain that knowledge.
- Gonna need to figure out how NumPy broadcasting works, though I already have an intuition of it that isn't what gives me a solid datapoint. KNOWING matters. That's ultimately truth.

'''

