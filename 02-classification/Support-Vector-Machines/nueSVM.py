from nue.models import SVM
from nue.preprocessing import x_y_split, csv_to_numpy

''' Pre-processing data '''

train_data = csv_to_numpy('data/dualSVM.csv')
test_data = csv_to_numpy('data/testdualSVM.csv')

X_train, Y_train = x_y_split(train_data, y_col = 'last')
X_test, Y_test = x_y_split(train_data, y_col = 'last')

''' Setting parameters '''

alpha = .0001
epochs = 10
verbose = True

''' Instantiating SVM '''

model = SVM()

''' Training and Testing the SVM '''

model.train(X_train, Y_train, verbose = verbose, epochs = epochs, alpha = alpha)
model.test(X_test, Y_test, verbose = verbose)