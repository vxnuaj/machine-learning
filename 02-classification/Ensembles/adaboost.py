'''
TODO

- I think should remodel to use gini index.

'''

import numpy as np
from nue.preprocessing import csv_to_numpy, x_y_split, train_test_split

class Stump:
    def __init__(self):
        self.polarity = 1
        self.feat_idx = None
        self.threshold = None
        self.alpha = None
        
    def predict(self, X):
        n_samples = X.shape[0]
        X_col = X[:, self.feat_idx]

        preds = np.ones(n_samples)
        if self.polarity == 1:
            preds[X_col < self.threshold] = 1 # if the polarity is 1, set the idxs in preds to be -1 for indices in X_col where values are less than the threshold
        else:
            preds[X_col >= self.threshold] = -1 # if the polarity is -1, set the idxs in preds to be -1 for the indices in X_col where value are greater than the thresholdh
        return preds
    
class AdaBoost:
    def __init__(self, verbose_train, verbose_test):
        self.verbose_train = verbose_train
        self.verbose_test = verbose_test
        
    def train(self, X_train, Y_train, n_stumps, seed = None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.n_stumps = n_stumps
        self.stumps = [] 
        
        n_samples, n_features = self.X_train.shape
        w = np.full(shape = n_samples, fill_value = (1 / n_samples))  # init weights
        
        for i in range(self.n_stumps): # for the number of stumps to be trained
            stump = Stump() # create a stump
            min_error = float('inf') # set the default error to the infinity
            for feat_idx in range(n_features): # for each feature in the dataset
                X_col = self.X_train[:, feat_idx] # get the column of the current feature being iterated on
                thresholds = np.unique(X_col) # get the unique threshold values in the feature column
                for thresh in thresholds: # for each unique threshold value in the column
                    p = 1 # polarity = 1
                    preds = np.ones(n_samples) # initial preds are all 1s
                    preds[X_col < thresh] = -1 # where the values are less than the threshold, set predictions to -1
                    
                    err = np.sum(w[self.Y_train.flatten() != preds.flatten()]) # compute the error term, sum of weights
                   
                    if err > .5: # if the error is greater than .5, error is 1 - error and flip the polarity
                        err = 1 - err 
                        p = -1
                   
                    # gets the feature split, threshold split, and polarity for the optimal split
                    if err < min_error: # if the error is less than the min_error, set the new min error to be the error
                        min_error = err
                        stump.polarity = p # get the current polarity
                        stump.feat_idx = feat_idx # get the best feature idx
                        stump.threshold = thresh # get the best threshold split

            preds = stump.predict(self.X_train) # get the predictions of the current stump
            stump.alpha = self._alpha(err) # compute the amount of say for the current stump
            w = self._update_weights(stump, preds, self.Y_train, w)  # compute the weight update for the current stump
            self.stumps.append(stump) # append the stump to the list of models in the ensemble

            if self.verbose_train:
                acc, loss = self._predict(self.X_train, self.Y_train)
                print(f"Stump: {i + 1} | Accuracy: {acc} | Loss: {loss}") 
            
    def test(self, X_test, Y_test = None):
        self.X_test = X_test
        self.Y_test = Y_test
        
        raw_preds = np.sum([stump.alpha * stump.predict(X_test) for stump in self.stumps], axis = 0)
        preds = np.sign(raw_preds)
        if self.Y_test.any():
            self.test_loss = self._exp_loss(self.Y_test, raw_preds) 
            self.test_acc = self._accuracy(self.Y_test, preds)
            if self.verbose_test: 
                print(f"\nTesting Accuracy: {self.test_acc}") 
                print(f"Testing Loss: {self.test_loss}")
        
        return preds

    def _alpha(self, err):
        eps = 1e-10
        return (.5) * np.log((1 - err) / (err + eps))

    def _update_weights(self, stump, preds, Y, w):
        w *= np.exp(-stump.alpha * preds.flatten() * Y.flatten())
        w /= np.sum(w)
        return w
        
    def _predict(self, X, Y):
        raw_preds = np.sum([stump.alpha * stump.predict(X) for stump in self.stumps], axis = 0)
        preds = np.sign(raw_preds)
        acc = self._accuracy(Y, preds)
        loss = self._exp_loss(Y, raw_preds)
        return acc, loss

    def _accuracy(self, Y, preds):
        return np.sum(Y.flatten() == preds.flatten()) / Y.size * 100
   
    def _exp_loss(self, Y, raw_preds):
        loss = np.mean(np.exp(- Y.flatten() * raw_preds.flatten()))    
        return loss

if __name__ == "__main__":
    data = csv_to_numpy("data/DesTreeData.csv")
    train, test = train_test_split(data, train_split = .8)
    X_train, Y_train = x_y_split(train, y_col = 'last')
    X_test, Y_test = x_y_split(test, y_col = 'last')
    Y_train = np.where(Y_train == 0, -1, 1)
    Y_test = np.where(Y_test == 0, -1, 1)
    
    verbose_train = True
    verbose_test = True
    n_stumps = 50
    seed = 1
    
    model = AdaBoost(verbose_train = verbose_train, verbose_test=verbose_test)
    model.train(X_train, Y_train, n_stumps = n_stumps, seed = seed)
    model.test(X_test, Y_test) 
