from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

lr = LinearRegression()
lrsgd = SGDRegressor(tol = None, max_iter = 500000, penalty=None, alpha = .001)

data = pd.read_csv('multiple_linear_regression_dataset.csv')
data = np.array(data)

X_train = data[:, :2]
Y_train = data[:, 2].reshape(-1, 1)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
Y_train = ss.fit_transform(Y_train)

lrsgd.fit(X_train, Y_train.ravel())

Y_pred = lrsgd.predict(X_train)

mse = mean_squared_error(Y_train, Y_pred)

print(f"Multiple Linear Regression with sklearn SGDRegressor")
print('weights', lrsgd.coef_)
print('bias',  lrsgd.intercept_)
print(f"MSE: {mse}")