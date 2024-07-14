import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

n_neighbors = 100
testing_size = 100

train_data = np.array(pd.read_csv('data/mnist_train.csv'))
test_data = np.array(pd.read_csv('data/mnist_test.csv'))

X_train = train_data[:, 1:785] / 255
Y_train = train_data[:, 0]

X_test = test_data[:testing_size, 1:785] / 255
Y_test = test_data[:testing_size, 0]

model = KNeighborsClassifier(n_neighbors = n_neighbors, metric = 'l2', algorithm='ball_tree')

model.fit(X_train, Y_train)
print(f"Model Fit")

print(f"Model Testing")
print(model.score(X_test, Y_test))