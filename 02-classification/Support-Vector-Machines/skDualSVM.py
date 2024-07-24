from sklearn.svm import SVC 
from sklearn.svm import SVC
from nue.preprocessing import x_y_split, csv_to_numpy

train_data = csv_to_numpy('data/dualSVM.csv')    
test_data = csv_to_numpy('data/testdualSVM.csv')

X_train, Y_train = x_y_split(train_data, y_col = 'last')
X_test, Y_test = x_y_split(test_data, y_col = 'last')

X_train, X_test = X_train.T, X_test.T
Y_train, Y_test = Y_train.flatten(), Y_test.flatten()

model = SVC(C = 0.0001, kernel = 'rbf', gamma = 'scale')

model2 = SVC(C = 0.0001, kernel = 'linear')

model.fit(X_train, Y_train)
print(f"Dual SVM: {model.score(X_test, Y_test) * 100}%")

model2.fit(X_train, Y_train)
print(f"Primal SVM (linear): {model2.score(X_test, Y_test) * 100}%")