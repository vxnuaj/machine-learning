import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


model = LogisticRegression()

data = np.array(pd.read_csv('data/iris.csv'))
train, test = train_test_split(data, train_size = .8, random_state=1)

X_train = train[:, :4]
Y_train = train[:, 4]

X_test = test[:, :4]
Y_test = test[:, 4]

model.fit(X_train, Y_train)
print(model.score(X_test, Y_test))
