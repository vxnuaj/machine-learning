import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model = LinearRegression()
ss = StandardScaler()

data = np.array(pd.read_csv('data/quad.csv'))
data = ss.fit_transform(data)

train, test = train_test_split(data, test_size = .2)

X_train = train[:, :2]
Y_train = train[:, 2]

cross_val = cross_val_score(model, X_train, Y_train, cv = 4, scoring = 'neg_mean_squared_error')
print(cross_val)