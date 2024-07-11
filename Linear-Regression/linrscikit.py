# SCIKIT IMPLEMENTATION FOR COMPARISON TO LINREGSELF.PY

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

data = pd.read_csv("./Data/random1.csv")
data = np.array(data)

x = data[:, 0]
y = data[:, 1]


reg = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1,1))

print("Scikit-learn slope:", reg.coef_[0][0])
print("Scikit-learn intercept:", reg.intercept_[0])