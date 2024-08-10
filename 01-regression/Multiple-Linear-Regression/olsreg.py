import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

data = pd.read_csv('../data/quad.csv')
data = np.array(data)

train, test = train_test_split(data, test_size = .3)

X_train = train[:, :2]
Y_train = train[:, 2].reshape(-1, 1)

ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)
Y_train = ss_train.fit_transform(Y_train)

X_test = test[:, :2]
Y_test = test[:, 2].reshape(-1, 1)

ss_test = StandardScaler()
X_test = ss_train.fit_transform(X_test)
Y_test = ss_train.fit_transform(Y_test)
eps = 1e-8

print(f'x {X_train.shape}')
print(f'y {Y_train.shape}')

Y_mean = np.mean(Y_train, axis = 0)
X_mean = np.mean(X_train, axis = 0)

print(f'xmean: {X_mean}')
print(f"ymean: {Y_mean}\n")

w= (np.sum((X_train - X_mean) * (Y_train -Y_mean), axis = 0) / np.sum(np.square(X_train - X_mean) + eps, axis = 0)) / 2
b = np.sum(Y_mean - w * X_mean) / 2

Y_pred = np.dot(w, X_test.T)

mse = mean_squared_error(Y_test, Y_pred)

standard_error_bias = np.std(Y_train) * (( 1 / len(Y_train)) + (np.square(np.mean(X_train)) / (np.sum(np.square(X_train - np.mean(X_train))))))
standard_error_weight = (np.var(Y_train) / np.sum(np.square(X_train - np.mean(X_train)), axis = 0 ))

print('weights', w)
print('bias',b)
print(mse)
print(f"Standard error bias: {standard_error_bias}")
print(f"Standard error weights: {standard_error_weight}")
