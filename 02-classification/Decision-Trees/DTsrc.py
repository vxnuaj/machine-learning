from dtree import DecisionTree
from nue.preprocessing import csv_to_numpy, train_test_split, x_y_split, label_encoding

''' Pre-processing data '''

data = csv_to_numpy('data/DesTreeData.csv')

train_data, test_data = train_test_split(data, train_split = .8)

X_train, Y_train = x_y_split(train_data, y_col = 'last')
X_test, Y_test = x_y_split(test_data, y_col = 'last')

X_train, Y_train = X_train.T, Y_train.T.astype(int)
X_test, Y_test = X_test.T, Y_test.T.astype(int)

''' Setting hyperparameters '''

max_depth = 1000
min_sample_split = 2

''' Instantiating model '''

model = DecisionTree(max_depth = max_depth, min_sample_split = min_sample_split, modality = 'entropy')

''' Training and testing the Decision Tree'''

model.fit(X_train, Y_train, alpha = 1, verbose = True)
model.predict(X_test, Y_test, verbose = True)