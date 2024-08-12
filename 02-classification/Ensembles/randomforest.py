import numpy as np
import pandas as pd
import time
from nue.preprocessing import x_y_split, train_test_split, csv_to_numpy

'''
building a random forest.

each decision tree selects a random subset of features to consider, denoted by a hyperparameter, let's

'''

class Node:
    def __init__(self, value = None, Y = None, right_node = None, left_node = None, threshold = None, feature = None):
        self.value = value
        self.Y = Y
        self.right_node = right_node
        self.left_node = left_node
        self.threshold = threshold
        self.feature = feature

    def _is_leaf(self):
        return self.value is not None # return True if self.value isn't None, otherwise if the Node has a value return False. Checks if it is a Leaf node or not.

class RandomTree:
    def __init__(self, verbose_train = False, n_extremely_randomized_feats = None):
        self.verbose_train = verbose_train
        self.n_extremely_randomized_feats = n_extremely_randomized_feats
        self.root = None # the model isn't trained if None
        self.n_leaf = 0 # number of leaf nodes. at init is 0

    def train(self, X_train, Y_train, min_node_samples = 2, max_depth = 100, max_features = 5, criterion = 'gini', alpha = 0, n_random_thresh = None): 
        self.X_train = X_train
        self.Y_train = Y_train
        self.min_node_samples = min_node_samples
        self.max_depth = max_depth
        self.max_features = max_features
        self.criterion = criterion
        self.alpha = alpha
        self.root = self._grow_tree(self.X_train, self.Y_train)
    
    def test(self, X_test, Y_test):
        self.X_test = X_test
        self.Y_test = Y_test
        self.preds = np.array([self._traverse(x) for x in X_test])
        self.accuracy = self._accuracy()

        print(f"Accuracy: {self.accuracy}%") 

        return self.preds

    def _grow_tree(self, X, Y, depth = 0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(Y))

        # if a node has no value to assign itself to
        if len(Y) == 0: 
            return Node(value = None) 
        
        # stopping criteria
        if (depth == self.max_depth or n_classes == 1 or n_samples < self.min_node_samples):
            leaf_val = self._most_common_label(Y)
            self.n_leaf += 1
            return Node(value = leaf_val, Y = Y)

        best_thresh, best_feat = self._best_split(X, Y)
        
        # pruning criteria
        if best_thresh is None or best_feat is None:
            leaf_val = self._most_common_label(Y)
            self.n_leaf += 1
            return Node(value = leaf_val, Y = Y)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        depth += 1

        if self.verbose_train:
            print(f"Tree Depth: {depth}")

        left_node = self._grow_tree(X[left_idxs], Y[left_idxs], depth = depth)
        right_node = self._grow_tree(X[right_idxs], Y[right_idxs], depth = depth)
        return Node(right_node = right_node, left_node = left_node, threshold = best_thresh, feature = best_feat, Y = Y)

    def _most_common_label(self, Y):
        labels, freqs = np.unique(Y.flatten(), return_counts = True)
        most_common_idx = np.argmax(freqs)
        return labels[most_common_idx]

    def _best_split(self, X, Y):
        n_samples, n_features = X.shape
        best_thresh = None
        best_feat = None
        best_gain = -1
        
        feat_idxs = np.random.randint(low = 0, high = X.shape[1], size = self.max_features)
        if self.n_extremely_randomized_feats:
            for feat_idx in feat_idxs:
                X_col = X[:, feat_idx]
                thresholds = np.unique(X_col)
                thresh_idxs =  np.random.randint(low = 0, high=X_col.size, size = self.n_extremely_randomized_feats)

                for thresh_idx in thresh_idxs:
                    thresh_val = X_col[thresh_idx] 
                    inf_gain = self._inf_gain(X_col, Y, thresh_val)
                    if inf_gain > best_gain:
                        best_gain = inf_gain
                        best_thresh = thresh_val
                        best_feat = feat_idx
            return best_thresh, best_feat

        for feat_idx in feat_idxs:
            X_col = X[:, feat_idx]
            thresh_vals = np.unique(X_col)
            for thresh_val in thresh_vals:
                inf_gain = self._inf_gain(X_col, Y, thresh_val)
                if inf_gain > best_gain:
                    best_gain = inf_gain
                    best_thresh = thresh_val
                    best_feat = feat_idx
        return best_thresh, best_feat

    def _inf_gain(self, X_col, Y, thresh):
        
        left_idxs, right_idxs = self._split(X_col, thresh)

        # if a left_idxs or right_idxs has no value to split upon, indicates that the given split with len == 0 is the worst possible split to go down.
        # prevents empty nodes with no values from being considered
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return -1000
         
        n = len(Y)
        n_l = len(left_idxs)
        n_r = len(right_idxs)

        if self.criterion == 'gini':
            parent_gini = self._gini(Y)
            left_gini, right_gini = self._gini(Y[left_idxs]), self._gini(Y[right_idxs])
            if self.alpha: 
                weighted_gini = ((n_l / n ) * left_gini + (n_r / n) * right_gini) + (self.alpha * np.abs((self.n_leaf)))
            else:
                weighted_gini = (n_l / n ) * left_gini + (n_r / n) * right_gini
            return parent_gini - weighted_gini
            
        elif self.criterion == 'entropy':
            parent_entropy = self._entropy(Y)
            left_ent, right_ent = self._entropy(Y[left_idxs]), self._entropy(Y[right_idxs])
            if self.alpha: 
                weighted_ent = ((n_l / n ) * left_ent + (n_r / n) * right_ent) + (self.alpha * np.abs((self.n_leaf)))
            else: 
                weighted_ent = (n_l / n ) * left_ent + (n_r / n) * right_ent
            return parent_entropy - weighted_ent

    def _split(self, X_col, thresh):
        left_idxs = np.argwhere(X_col < thresh).flatten()
        right_idxs = np.argwhere(X_col >= thresh).flatten()
        return left_idxs, right_idxs

    def _gini(self, Y):
        _, freqs = np.unique(Y.flatten(), return_counts = True)
        probs = freqs / Y.size
        gini = 1 - np.sum(np.square(probs))
        return gini

    def _entropy(self, Y):
        _, freqs = np.unique(Y.flatten(), return_counts =True)
        probs = freqs / Y.size
        entropy = - np.sum(probs * np.log(probs))
        return entropy

    def _accuracy(self):
        self.acc = np.sum(self.preds.flatten() == self.Y_test.flatten()) / self.Y_test.size * 100
        return self.acc

    def _traverse(self, x):
        node = self.root
        while not node._is_leaf():
            if x[node.feature] >= node.threshold:
                node = node.right_node
            elif x[node.feature] < node.threshold:
                node = node.left_node
        return node.value

class RandomForest:

    def __init__(self, verbose_train = False, n_extremely_randomized_feats = None):
        self.verbose_train = verbose_train
        self.n_extremely_randomized_feats = n_extremely_randomized_feats

    def train(self, X_train, Y_train, max_features = 5, n_bootstraps = 10, rtree_dict = None, alpha_range = None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.max_features = max_features
        self.n_bootstraps = n_bootstraps
        self.rtree_dict = rtree_dict
        self.alpha_range = alpha_range
        self.models = []

        self._get_dicts()
        
        for i in range(n_bootstraps):
            b_idx = self._bootstrap_idx()
            model = RandomTree(n_extremely_randomized_feats = self.n_extremely_randomized_feats, **self._init_dict)
            print(f"Training Tree #{i}")
            model.train(X_train[b_idx],Y_train[b_idx], max_features = self.max_features, **self._train_dict)
            self.models.append(model)

        print(f"\nAll {i} Trees have finished Training.\n")

    def test(self, X_test, Y_test):
        self.X_test = X_test
        self.Y_test = Y_test
        self.preds = []

        for model in self.models:
            preds = model.test(self.X_test, self.Y_test)
            self.preds.append(preds)
           
        self._get_preds()        
        self._accuracy()
        
        print(f"\nFinal Forest Accuracy: {self.accuracy}%")

    def _get_dicts(self):
        self._init_dict = {k:v for k,v in self.rtree_dict.items() if k in ['verbose_train']}
        self._train_dict = {k:v for k,v in self.rtree_dict.items() if k in ['min_node_samples', 'max_depth', 'criterion']}

    def _bootstrap_idx(self):
        n_samples = self.Y_train.size
        b_idx = np.random.randint(low = 0, high = n_samples, size = n_samples)
        return b_idx

    def _get_preds(self):
        self.preds = np.array(self.preds)
        self.preds = np.apply_along_axis(self._most_common_label, axis = 0, arr = self.preds)

    def _most_common_label(self, preds):
        pred, freqs = np.unique(preds, return_counts = True)
        most_common_idx = np.argmax(freqs)
        return pred[most_common_idx]

    def _accuracy(self):
        self.accuracy = np.sum(self.preds.flatten() == self.Y_test.flatten()) / self.Y_test.size * 100
        
    @property
    def max_features(self):
        return self._max_features

    @max_features.setter
    def max_features(self, max_features):
        assert 0 < max_features < self.X_train.shape[1], "max_features can't be or exceed the total number of features in X_train."
        self._max_features = max_features

if __name__ == "__main__":

    data = csv_to_numpy('data/DesTreeData.csv')
    train, test = train_test_split(data, train_split = .8)
    X_train, Y_train = x_y_split(train, y_col = 'last')
    X_test, Y_test = x_y_split(test, y_col = 'last') # dataset has 9 features

    start_time = time.time()

    verbose_train = True
    n_extremely_randomized_feats = 50
    n_bootstraps = 5
    rtree_dict = {
            'verbose_train': False,
            'min_node_samples': 2,
            'max_depth': 100, 
            'criterion': 'entropy'
    }

    model = RandomForest(verbose_train = verbose_train, n_extremely_randomized_feats = n_extremely_randomized_feats)
    model.train(X_train, Y_train, max_features = 5, n_bootstraps = 5, rtree_dict = rtree_dict)
    model.test(X_test, Y_test)
    print(model.n_extremely_randomized_feats)
    end_time = time.time()
    execution = end_time - start_time
    print(f"Took: {execution} seconds")
    '''    model = RandomTree(verbose_train = verbose_train)
    model.train(X_train = X_train, Y_train = Y_train, alpha = 0, max_depth = 1000, min_node_samples = 2)
    model.test(X_test = X_test, Y_test = Y_test)
    '''
