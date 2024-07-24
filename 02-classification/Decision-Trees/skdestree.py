from sklearn.tree import DecisionTreeClassifier
from nue.preprocessing import csv_to_numpy, train_test_split, x_y_split, label_encoding

''' Pre-processing data '''

data = csv_to_numpy('data/DesTreeData.csv')

train_data, test_data = train_test_split(data, train_split = .8)

X_train, Y_train = x_y_split(train_data, y_col = 'last')
X_test, Y_test = x_y_split(test_data, y_col = 'last')


X_train, Y_train = X_train.T, Y_train.T.astype(int)
X_test, Y_test = X_test.T, Y_test.T.astype(int)

model = DecisionTreeClassifier(criterion = 'gini', splitter = 'best', max_depth = 1000, min_samples_split = 2)
model.fit(X_train, Y_train)

print(f"Accuracy: {model.score(X_test, Y_test)}")


