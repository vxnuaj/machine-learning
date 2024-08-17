import numpy as np
from nue.preprocessing import csv_to_numpy, x_y_split, train_test_split

'''
TODO: 

- [ ] build gradient boosting / decision tree with classification
- [ ] build graidnet boosting / decision tree with regression

'''


class Node:
    def __init__(self, value = None, Y = None, feature = None, threshold = None, left_node = None, right_node = None, is_root = False, depth = 0, is_base = False):
        self.value = value
        self.Y = Y
        self.feature = feature
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.is_base = is_base

    def _is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, verbose_train, verbose_test):
        self.verbose_train = verbose_train
        self.verbose_test = verbose_test
        self.root = None
        self.n_leaf = 0

    def train(self, X_train, Y_train, modality = 'classification', criterion = 'gini', max_depth = 8, min_node_samples = 2):
        self.X_train = X_train
        self.Y_train = Y_train
        self.modality = modality
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_node_samples = min_node_samples
        '''
        note that for regression, the criterion has to be mse or mae. 
        for classificaiton, can be entropy or gini
        '''
        self.root = self._grow_tree(self.X_train, self.Y_train) 
      
    def test(self, X_test, Y_test = None):
        self.X_test = X_test
        self.Y_test = Y_test

        self.preds = np.array([self._traverse(x) for x in self.X_test])
        if self.Y_test.any():
            acc = self._accuracy(self.Y_test, self.preds)
            if self.verbose_test:
                print(f"\nAccuracy: {acc}%")

    def _grow_tree(self, X, Y, depth = 0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(Y))
        
        if (depth == self.max_depth or n_classes == 1 or n_samples < self.min_node_samples):
            leaf_val = self._most_common_labels(Y)
            self.n_leaf += 1
            return Node(value = leaf_val, Y=Y)

        best_feat, best_thresh = self._best_split_classification(X, Y)

        if best_thresh is None or best_feat is None:
            leaf_val = self._most_common_labels(Y)
            self.n_leaf += 1
            return Node(value = leaf_val, Y = Y)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        if self.verbose_train:
            print(f"Tree Depth: {depth}")
        
        depth += 1

        left_node = self._grow_tree(X[left_idxs], Y[left_idxs], depth = depth)
        right_node = self._grow_tree(X[right_idxs], Y[right_idxs], depth = depth)


        return Node(Y=Y, left_node = left_node, right_node = right_node, threshold = best_thresh, feature = best_feat)
             
    def _most_common_labels(self, Y):
        labels, counts = np.unique(Y.flatten(), return_counts = True)
        idx = np.argmax(counts)
        return labels[idx]

    def _best_split_classification(self, X, Y):
        n_samples, n_features = X.shape
        best_thresh, best_feat = None, None
        best_gain = -1000

        for feat_idx in range(n_features):
            X_col = X[:, feat_idx]
            thresholds = np.unique(X_col)
            for thresh in thresholds:
                information_gain = self._information_gain(X_col, Y, thresh)
                if information_gain > best_gain:
                    best_gain = information_gain
                    best_feat = feat_idx
                    best_thresh = thresh
        return best_feat, best_thresh

    
    def _information_gain(self, X_col, Y, thresh):
        left_idxs, right_idxs = self._split(X_col, thresh)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
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
            left_ent, right_ent  = self._entropy(Y[left_idxs]), self._entropy(Y[right_idxs])
            weighted_ent = ((n_l / n) * left_ent + (n_r / n) * right_ent)             
            return parent_ent - weighted_ent

    def _split(self, X_col, thresh):
        left_idxs = np.argwhere(X_col < thresh).flatten()
        right_idxs = np.argwhere(X_col >= thresh).flatten()
        return left_idxs, right_idxs

    def _gini(self, Y):
        labels, counts = np.unique(Y.flatten(), return_counts = True)
        probs = counts / Y.size
        return 1 - np.sum(np.square(probs))
   
    def _entropy(self, Y):
        labels, counts = np.unique(Y.flatten(), return_counts = True)
        probs = counts/ Y.size
        return - np.sum(probs * np.log(probs))

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

class GradientBoost:
    def __init__(self, verbose_train, verbose_test):
        self.verbose_train = verbose_train
        self.verbose_test = verbose_test

    def train(self, X_train, Y_train, modality = 'classification'):
        self.X_train = X_train
        self.Y_train = Y_train
        self.modality = modality

        if self.modality == 'classification':
            self._train_classification(self.X_train, self.Y_train)
    
    def _train_classification(self, X, Y):
        init_probs = self._log_odds(Y)
        init_residuals = Y_train - init_probs
        raw_output = self._transform_log_odds(init_residuals)

    def _log_odds(self, Y):
        labels, freqs = np.unique(Y, return_counts = True)
        probs = freqs / Y.size
        return np.log(probs / (1 - probs))

    def _transform_residuals(self, residuals, probs):
        return (np.sum(residuals) / np.sum(probs * (1 - probs))

    def test(self, X_test, Y_test):
        pass


if __name__ == "__main__":
    data = csv_to_numpy("data/DesTreeData.csv")
    train, test = train_test_split(data, train_split = .8)
    X_train, Y_train = x_y_split(train, y_col = 'last')
    X_test, Y_test = x_y_split(test, y_col = 'last')

    


    ''' 
    --- VALIDATING DECISION TREE CLASSIFICATION ---    
    verbose_train = True
    verbose_test = True
    modality = 'classification'
    criterion = 'gini'
    max_depth = 16

    model = DecisionTree(verbose_train = verbose_train, verbose_test = verbose_test)
    model.train(X_train, Y_train, modality = modality, criterion = criterion, max_depth = max_depth)
    model.test(X_train, Y_train)
    '''
