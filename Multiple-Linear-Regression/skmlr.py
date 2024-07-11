from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

data = np.array(pd.read_csv('data/quad.csv'))

train, test = train_test_split(data, test_size = .3, random_state=1)

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

model = LinearRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)


standard_error_bias = np.std(Y_train) * (( 1 / len(Y_train)) + (np.square(np.mean(X_train)) / (np.sum(np.square(X_train - np.mean(X_train))))))
standard_error_weight = (np.var(Y_train) / np.sum(np.square(X_train - np.mean(X_train)), axis = 0 ))

print(f"Weights: {model.coef_}")
print(f"Bias: {model.intercept_}")
print(f"MSE: {mse}")
print(f"R2: {r2}")
print(f"Standard error bias: {standard_error_bias}")
print(f"Standard error weights: {standard_error_weight}")

