import numpy as np
from termcolor import colored
from nue.preprocessing import csv_to_numpy, x_y_split, train_test_split

class AdaBoost:
    def __init__(self, verbose_train, verbose_test):
        self.verbose_train = verbose_train
        self.verbose_test = verbose_test
        self.stumps = []

    def train(self, X_train, Y_train, n_stumps, criterion = 'gini'):
        self.X_train = X_train
        self.Y_train = Y_train
        self.n_stumps = n_stumps
        self.criterion = criterion
        self.polarity = 1
        self._train_stumps()

    def test(self, X_test, Y_test):
        self.X_test = X_test
        self.Y_test = Y_test
        raw_preds = np.sum([stump.test(self.X_test, self.Y_test) * alpha for stump, alpha in self.stumps], axis = 0)
        preds = np.sign(raw_preds)
        self.test_loss = self._exp_loss(self.Y_test, raw_preds)
        self.test_acc = self._accuracy(self.Y_test, preds)

        if self.verbose_test:
            print(colored("\nTESTING ADABOOST:", 'green', attrs = ['bold', 'underline']), f"\nACCURACY: {self.test_acc}\nLOSS: {self.test_loss}")

        return preds

    def _train_stumps(self):
        n_samples, n_features = self.X_train.shape
        weights = np.full(shape = n_samples, fill_value = (1 / n_samples))

        for i in range(self.n_stumps):
            stump = Stump()
            stump.train(self.X_train, self.Y_train, self.criterion)
            preds = stump.test(self.X_train, self.Y_train)
            err = self._get_err(preds, self.Y_train, weights, stump)
            alpha = self._get_alpha(err)
            weights = self._update_weights(weights, alpha, preds) 
            self.stumps.append((stump, alpha))

            if self.verbose_train:
                acc, loss = self._train_metric()
                print(f"STUMP #{i} | ACCURACY: {acc} | LOSS: {loss}")

    def _get_err(self, preds, Y, weights, stump):
        err = np.sum(weights[Y.flatten() != preds.flatten()])
        if err > .5:
            err = 1 - err
            stump.polarity = -1
        return err
    
    def _get_alpha(self, err):
        eps = 1e-10
        return (.5) * np.log((1 - err) / (err + eps))

    def _update_weights(self, w, alpha, preds):
        w *= np.exp(-alpha * preds * self.Y_train.flatten())
        w /= np.sum(w)
        return w

    def _train_metric(self):
        raw_preds = np.sum([stump.test(self.X_train, self.Y_train) * alpha for stump, alpha in self.stumps], axis = 0)
        preds = np.sign(raw_preds)
        loss = self._exp_loss(self.Y_train, raw_preds)
        acc = self._accuracy(self.Y_train, preds)
        return acc, loss

    def _accuracy(self, Y, preds):
        return np.mean(Y.flatten() == preds.flatten()) * 100

    def _exp_loss(self, Y, preds):
        return np.mean(np.exp(- Y.flatten() * preds.flatten()))

class Stump:
    def __init__(self, verbose_train = False, verbose_test = False):
        self.verbose_train = verbose_train 
        self.verbose_test = verbose_test

    def train(self, X_train, Y_train, criterion = 'gini'):
        self.X_train = X_train
        self.Y_train = Y_train
        self.criterion = criterion
        self.max_depth = 1
        self.polarity = 1
        self.root = self._grow_stump(self.X_train, self.Y_train)

    def test(self, X_test, Y_test):
        self.X_test = X_test
        self.Y_test = Y_test
        self.preds = np.array([self._traverse(x) for x in self.X_test])
        if self.polarity == -1:
            self.preds = -self.preds

        if self.verbose_test:
            acc = self._accuracy(self.Y_test, self.preds)
            print(f"TESTING ACCURACY: {acc}")
        return self.preds

    def _grow_stump(self, X, Y, depth = 0):
        #n_samples, n_features = X.shape

        if depth == self.max_depth:
            leaf_val = self._most_common_label(Y)
            return _Node(value = leaf_val)

        best_feat, best_thresh = self._best_split(X, Y)

        if best_feat is None or best_thresh is None:
            leaf_val = self._most_common_label(Y)
            return _Node(value = leaf_val)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        depth += 1
        if self.verbose_train:
            print(f"Tree Depth: {depth}")
        left_node = self._grow_stump(X[left_idxs], Y[left_idxs], depth = depth)
        right_node = self._grow_stump(X[right_idxs], Y[right_idxs], depth = depth)
        return _Node(Y = Y, feature = best_feat, threshold = best_thresh, left_node = left_node, right_node = right_node) 

    def _best_split(self, X, Y):
        n_samples, n_features = X.shape
        best_gain = float('-inf')
        best_thresh, best_feat = None, None

        for feat_idx in range(n_features):
            X_col = X[:, feat_idx]
            thresholds = np.unique(X_col)
            for thresh in thresholds:
                inf_gain = self._inf_gain(X_col, Y, thresh)
                if inf_gain > best_gain:
                    best_gain = inf_gain
                    best_feat = feat_idx
                    best_thresh = thresh

        return best_feat, best_thresh

    def _split(self, X_col, thresh):
        left_idxs = np.argwhere(X_col < thresh).flatten()
        right_idxs = np.argwhere(X_col >= thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, Y):
        labels, counts = np.unique(Y, return_counts = True)
        idx = np.argmax(counts)
        return labels[idx]

    def _inf_gain(self, X_col, Y, thresh):
        left_idxs, right_idxs = self._split(X_col, thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return -1000

        n = len(Y)
        n_l = len(left_idxs)
        n_r = len(right_idxs)

        if self.criterion == 'gini':
            parent_gini = self._gini(Y)
            left_gini, right_gini = self._gini(Y[left_idxs]), self._gini(Y[right_idxs])
            weighted_gini = (n_l / n) * left_gini + (n_r / n) * right_gini
            return parent_gini - weighted_gini
        elif self.criterion == 'entropy':
            parent_ent = self._entropy(Y)
            left_ent, right_ent = self._entropy(Y[left_idxs]), self._entropy(Y[right_idxs])
            weighted_ent = (n_l / n) * left_ent + (n_r / n) * right_ent
            return parent_ent - weighted_ent

    def _gini(self, Y):
        labels, counts = np.unique(Y, return_counts = True)
        probs = counts / Y.size
        return 1 - np.sum(np.square(probs))
    
    def _entropy(self, Y):
        labels, counts = np.unique(Y, return_counts = True)
        probs = counts / Y.size
        eps = 1e-10
        return - np.sum(probs * np.log(probs + eps))

    def _accuracy(self, Y, preds):
        return np.sum(Y.flatten() == preds.flatten()) / Y.size * 100

    def _traverse(self, x):
        node = self.root
        while not node._is_leaf():
            if x[node.feature] < node.threshold:
                node = node.left_node
            elif x[node.feature] >= node.threshold:
                node = node.right_node

        return node.value

class _Node:
    def __init__(self, value = None, Y = None, feature = None, threshold = None, left_node = None, right_node = None):
        self.value = value
        self.Y = Y
        self.feature = feature
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node

    def _is_leaf(self):
        return self.value is not None

if __name__ == "__main__":
    data = csv_to_numpy("data/DesTreeData.csv")
    train, test = train_test_split(data, train_split = .8)
    X_train, Y_train = x_y_split(train, y_col = 'last')
    X_test, Y_test = x_y_split(test, y_col = 'last')
    Y_train = np.where(Y_train == 0, -1, 1)
    Y_test = np.where(Y_test == 0, -1, 1)

    verbose_train = True
    verbose_test = True
    n_stumps = 5
    criterion = 'entropy'

    model = AdaBoost(verbose_train = verbose_train, verbose_test = verbose_test)
    model.train(X_train, Y_train, n_stumps = n_stumps, criterion = criterion)
    model.test(X_test, Y_test)

    '''
    VALIDATING STUMP

    verbose_train = True
    verbose_test = True
    criterion = 'entropy'

    model = Stump(verbose_test = verbose_test, verbose_train = verbose_train)
    model.train(X_train, Y_train, criterion = criterion)
    model.test(X_train, Y_train)'''
