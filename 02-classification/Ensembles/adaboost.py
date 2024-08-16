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
            preds[X_col < self.threshold] = -1
        else:
            preds[X_col >= self.threshold] = -1
        return preds
    
class AdaBoost:
    def __init__(self, verbose_train, verbose_test):
        self.verbose_train = verbose_train
        self.verbose_test = verbose_test
        
    def train(self, X_train, Y_train, n_stumps, seed = None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.n_stumps = n_stumps
        self.rng = np.random.default_rng(seed = seed)
        self.stumps = [] 
        
        n_samples, n_features = self.X_train.shape
        w = np.full(shape = n_samples, fill_value = (1 / n_samples)) 
        
        for i in range(self.n_stumps):
            stump = Stump()
            min_error = float('inf')
            for feat_idx in range(n_features):
                X_col = self.X_train[:, feat_idx]
                thresholds = np.unique(X_col)
                for thresh in thresholds:
                    p = 1
                    preds = np.ones(n_samples)
                    preds[X_col < thresh] = -1
                    
                    err = np.sum(w[self.Y_train.flatten() != preds.flatten()])
                   
                    if err > .5:
                        err = 1 - err 
                        p = -1
                    
                    if err < min_error:
                        min_error = err
                        stump.polarity = p
                        stump.feat_idx = feat_idx
                        stump.threshold = thresh

            preds = stump.predict(self.X_train)
            stump.alpha = self._alpha(err)
            w = self._update_weights(stump, preds, self.Y_train, w) 
            self.stumps.append(stump)      

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

    def _update_samples(self, X, Y, w):
        n_samples = Y.size
        idxs = self.rng.choice(n_samples, size = n_samples, replace = True, p = w)
        X, Y= X[idxs], Y[idxs]
        return X, Y
    
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
    n_stumps = 1000
    
    model = AdaBoost(verbose_train = verbose_train, verbose_test=verbose_train)
    model.train(X_train, Y_train, n_stumps = n_stumps)
    model.test(X_test, Y_test) 