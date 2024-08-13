import numpy as np
from nue.preprocessing import x_y_split, train_test_split, csv_to_numpy

class _Node:
    def __init__(self, value = None, threshold = None, feature = None, right_node = None, left_node = None):
        self.threshold = threshold
        self.feature = feature
        self.value = value
        self.right_node = right_node
        self.left_node = left_node
    
    def _is_leaf(self):
        return self.value is not None

class _TreeStump:
    def __init__(self, verbose_train, verbose_test):
        self.verbose_train = verbose_train
        self.verbose_test = verbose_test
        self.max_depth = 1
        self.root = None # none if stump is not trained

    def train(self, X_train, Y_train, weights, criterion = 'gini'):
        self.X_train = X_train
        self.Y_train = Y_train
        self.weights = weights
        self.criterion = criterion
        self.root = self._grow_stump(self.X_train, self.Y_train)
        self.train_preds = np.array([self._t_traverse(x) for x in self.X_train]) 
        self.train_accuracy = self._accuracy(self.Y_train, self.train_preds)
        self.error, self.say = self._get_error_say()
        
        return self.preds, self.error, self.say

    '''def test(self, X_test, Y_test):
        self.X_test = X_test
        self.Y_test = Y_test
        self.preds = np.array([self._traverse(x) for x in X_test])
        self.accuracy = self._accuracy(Y, self.preds)
    '''
    def _grow_stump(self, X, Y, depth = 0):
        # depth stopping criterion
        if depth == 1:
            leaf_val = self._most_common_label(Y)
            return _Node(value = leaf_val)
        best_thresh, best_feat = self._best_split(X, Y)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        depth += 1
        left_node = self._grow_stump(X[left_idxs], Y[left_idxs], depth = depth)
        right_node = self._grow_stump(X[right_idxs], Y[right_idxs], depth = depth)
        return _Node(right_node = right_node, left_node = left_node, threshold = best_thresh, feature = best_feat)
   
    def _best_split(self, X, Y):
        n_samples, n_features = X.shape
        best_feat, best_thresh = None, None
        best_gain = -1000
        for feat_idx in range(n_features):
            X_col = X[:, feat_idx]
            thresholds = np.unique(X_col)
            for thresh in thresholds:
                inf_gain = self._inf_gain(X_col, Y, thresh)
                if inf_gain > best_gain:
                    best_gain = inf_gain
                    best_feat = feat_idx
                    best_thresh = thresh
        return best_thresh, best_feat
    
    def _inf_gain(self, X_col, Y, thresh):
        left_idxs, right_idxs = self._split(X_col, thresh)
        n = len(Y)
        n_l = len(left_idxs)
        n_r = len(right_idxs)
        if self.criterion == 'gini':
            parent_gini = self._gini(Y)
            left_gini, right_gini = self._gini(Y[left_idxs]), self._gini(Y[right_idxs])
            weighted_gini = (n_l / n) * left_gini + (n_r / n) * right_gini
            inf_gain = parent_gini - weighted_gini
            return inf_gain
        elif self.criterion == 'entropy':
            parent_ent = self._entropy(Y)
            left_ent, right_ent = self._entropy(Y[left_idxs]), self._entropy(Y[right_idxs])
            weighted_ent = (n_l / n) * left_ent + (n_r / n) * right_ent
            inf_gain = parent_ent - weighted_ent
            return inf_gain

    def _gini(self, Y):
        labels, freqs = np.unique(Y.flatten(), return_counts = True)
        probs = freqs / len(Y)
        gini = 1 - np.sum(np.square(probs))
        return gini

    def _entropy(self, Y):
        labels, freqs = np.unique(Y.flatten(), return_counts = True)
        probs = freqs / len(Y)
        eps = 1e-10
        ent = - np.sum(probs * np.log(probs + eps))
        return ent

    def _split(self, X_col, thresh):
        left_idxs = np.argwhere(X_col < thresh).flatten()
        right_idxs = np.argwhere(X_col >= thresh).flatten()
        return left_idxs, right_idxs
    
    def _most_common_label(self, Y):
        labels, freqs = np.unique(Y.flatten(), return_counts = True)
        most_common_idx = np.argmax(freqs)
        return labels[most_common_idx]

    def _accuracy(self, Y, preds):
        return np.sum(Y == preds) / Y.size * 100

    def _get_error_say(self):
        incorrect_idxs = np.argwhere(self.Y_train != self.train_preds)
        error = np.sum(self.weights[incorrect_idxs])
        say = (1/2) * np.log((1 - error) / (error))
        return error, say

    def _t_traverse(self):
        node = self.root
        while node._is_leaf():
            if x[:, node.feature] >= node.threshold:
                node = node.right_node
            elif x[:, node.feature] < node.threshold:
                node = node.left_node
        return node.value

    def _p_traverse(self):
        # the traversal function for conducting testing / inference on the model
        pass

class AdaBoost:
    def __init__(self, verbose_train, verbose_test):
        self.verbose_train = verbose_train
        self.verbose_test = verbose_test

    def train(self, X_train, Y_train, n_stump, stump_dict, criterion = 'gini'):
        self.X_train = X_train
        self.Y_train = Y_train
        self.n_stump = n_stump
        self.criterion = criterion

        self._get_dicts(stump_dict) 
        self._optim()

    def _optim(self):
        self._models = []
        w = np.array(shape = (self.Y_train.shape[0] ,1), fill_value = 1 / self.Y_train.shape[0])
        X, Y = self.X_train, self.Y_train
        for stump in range(self.n_stump):
            model = _TreeStump(**self._init_dict)
            preds, error, say = model.train(X, Y, weights = w, criterion = self.criterion)
            loss = self._exp_loss(Y, preds) 
            w = self._weight_update(w, Y, preds, say)
            X, Y = self._draw_samples(w)
            self._models.append(model)

# -- left off on constructing the optim -- trying to figure out how to draw samples based on weights.

    def _draw_samples(self, w = None):
        
            
    def _weight_update(self, w, Y, preds, say):
        w *= np.exp( - say * preds * Y)
        w /= np.sum(w)
        return w

    def _exp_loss(self, Y, preds):
        loss = np.sum(np.exp(- y * preds))
        return loss

    def _get_dicts(self, stump_dict):
        self._init_dict = {k:v for k, v in stump_dict.items() if k in ['verbose_train', 'verbose_test']}
        '''self._train_dict = {k:v for k, v in stump_dict.items() if k in ['']}
        self._test_dict = {k:v for k, v in stump_dict.items() if k in ['']]}'''
        

if __name__ == "__main__":

    data = csv_to_numpy("data/DesTreeData.csv")
    train, test = train_test_split(data, train_split = .8)
    X_train, Y_train = x_y_split(train, y_col = 'last')
    X_test, Y_test = x_y_split(test, y_col = 'last')

    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
