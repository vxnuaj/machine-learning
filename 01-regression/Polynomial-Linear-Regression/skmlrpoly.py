from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv('../data/quad.csv')
    data = np.array(data)

    X_train = data[:, :2]
    Y_train = data[:, 2].reshape(-1, 1)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    Y_train = ss.fit_transform(Y_train)
    
    poly = PolynomialFeatures(degree = 2)
    
    model = make_pipeline(poly, LinearRegression())
    
    model.fit(X_train, Y_train)
    
    Y_pred = model.predict(X_train)
    mse = mean_squared_error(Y_train, Y_pred)
    r2 = r2_score(Y_train, Y_pred)
   
    
    print(f'Muliple polynomial LR with sklearn LinearRegression')
    print('weights', model.steps[1][1].coef_)
    print('bias',  model.steps[1][1].intercept_)
    print(f"MSE: {mse}")
    print(f"R2: {r2}")