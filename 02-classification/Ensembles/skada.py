import numpy as np
from nue.preprocessing import csv_to_numpy, x_y_split, train_test_split
from sklearn.ensemble import AdaBoostClassifier

print('preprocessing')
data = csv_to_numpy("data/DesTreeData.csv")
train, test = train_test_split(data, train_split = .8)
X_train, Y_train = x_y_split(train, y_col = 'last')
X_test, Y_test = x_y_split(test, y_col = 'last')
Y_train = np.where(Y_train == 0, -1, 1)
Y_test = np.where(Y_test == 0, -1, 1)

model = AdaBoostClassifier(random_state = 1, algorithm='SAMME')
print('training model')
model.fit(X_train, Y_train.flatten())
score = model.score(X_test, Y_test.flatten())

print(score)
