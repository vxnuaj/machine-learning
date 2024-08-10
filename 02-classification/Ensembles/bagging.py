import numpy as np
from nue.preprocessing import x_y_split, csv_to_numpy, train_test_split

class Node:
    def __init__(self, value = None, Y = None, left_node = None, right_node = None, threshold = None, feature = None):
        self.value = value
        self.Y = Y
        self.left_node = left_node
        self.right_node = right_node
        self.threshold = threshold
        self.feature = feature
    
    def _is_leaf(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, max_depth, min_node_samples, verbose_train = False, verbose_test = False):
        self.max_depth = max_depth
        self.min_node_samples = min_node_samples
        self.root = None # if the tree isn't trained, the root is None
        self.n_leaf = 0  # no leaf nodes if the tree isn't trained
        self.verbose_train = verbose_train
        self.verbose_test = verbose_test
       
        
    def train(self, X_train, Y_train, alpha = .1, modality = 'entropy'):
        self.X_train = X_train
        self.Y_train = Y_train
        self.alpha = alpha
        self.criterion = modality
        
        self.root = self._grow_tree(self.X_train, self.Y_train) 
      
    def test(self, X_test, Y_test):
        self.X_test = X_test
        self.Y_test = Y_test
        self.preds = np.array([self._traverse(x) for x in self.X_test])
        self.accuracy = self._accuracy(Y_test, self.preds) 
   
        if self.verbose_test:
            print(f"Accuracy: {self.accuracy}%")
   
        return self.preds

    def _grow_tree(self, X, Y, depth = 0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(Y))
        
        if (depth == self.max_depth or n_classes == 1 or n_samples < self.min_node_samples):
            leaf_val = self._most_common_label(Y)
            self.n_leaf += 1
            return Node(value = leaf_val, Y = Y)
   
        best_feat, best_thresh = self._best_split(X, Y)
        
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
   
        return Node(Y=Y, left_node = left_node, right_node = right_node, threshold = best_thresh, feature = best_feat)
    
    def _most_common_label(self, Y):
        labels, freqs = np.unique(Y.flatten(), return_counts = True)
        most_common_label = np.argmax(freqs) 
        return labels[most_common_label]
    
    def _best_split(self, X, Y):
        n_samples, n_features = X.shape
        best_thresh, best_feat = None, None 
        best_gain = -1 
         
        for feat in range(n_features):
            X_col = X[:, feat]
            thresholds = np.unique(X_col) 
            for thresh in thresholds:
                information_gain = self._information_gain(X_col, Y, thresh)

                if information_gain > best_gain:
                    best_feat = feat
                    best_thresh = thresh
                    best_gain = information_gain
                    
        return best_feat, best_thresh 
                
    def _information_gain(self, X_col, Y, thresh):
        left_idxs, right_idxs = self._split(X_col, thresh) 
        n = len(Y) 
        n_l = len(left_idxs)
        n_r = len(right_idxs)
         
        if self.criterion == 'gini':
            parent_gini = self._gini(Y)
            left_gini, right_gini = self._gini(Y[left_idxs]), self._gini(Y[right_idxs])
            if self.alpha:
                weighted_gini = ((n_l / n) * left_gini + (n_r / n) * right_gini) + (self.alpha * np.abs(self.n_leaf))
            else:
                weighted_gini = (n_l / n) * left_gini + (n_r / n) * right_gini
            information_gain = parent_gini - weighted_gini 
        elif self.criterion == 'entropy': 
            parent_ent = self._entropy(Y)
            left_ent, right_ent  = self._entropy(Y[left_idxs]), self._entropy(Y[right_idxs])
            if self.alpha:
                weighted_ent = ((n_l / n) * left_ent + (n_r / n) * right_ent) + (self.alpha * np.abs(self.n_leaf))
            else:
                weighted_ent = ((n_l / n) * left_ent + (n_r / n) * right_ent)             
            information_gain = parent_ent - weighted_ent 
        return information_gain
                
    def _split(self, X_col, thresh):
        left_idxs = np.argwhere(X_col < thresh).flatten()
        right_idxs = np.argwhere(X_col >= thresh).flatten()
        return left_idxs, right_idxs
        
    def _entropy(self, Y):
        _, freqs = np.unique(Y.flatten(), return_counts = True)
        probs = freqs / Y.size
        ent = - np.sum(probs * np.log(probs))
        return ent 
        
    def _gini(self, Y):
        _, freqs = np.unique(Y.flatten(), return_counts = True)
        probs = freqs / Y.size
        gini = 1 - np.sum(np.square(probs))
        return gini 
   
    def _traverse(self, x):
        node = self.root
   
        while not node._is_leaf(): 
            if x[node.feature] < node.threshold: 
                node = node.left_node
            elif x[node.feature] >= node.threshold:
                node = node.right_node

        return node.value
    
    def _accuracy(self, Y, preds):
        acc = np.sum(Y.flatten() == preds.flatten()) / Y.size * 100
        return acc

class BaggedTrees:
    def __init__(self, verbose_train = False, verbose_test = False):
        self.verbose_train = verbose_train
        self.verbose_test = verbose_test
        self._preds = []
        
    def train(self, X_train, Y_train, n_bootstrap, dtree_dict, alpha_range:tuple = None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.n_bootstrap = n_bootstrap
        self.dtree_dict = dtree_dict
        self.alpha_range = alpha_range
        self.models = []

        for i in range(n_bootstrap):
            print(f"Training Tree {i}")
            X_bootstrap, Y_bootstrap = self._bootstrap_samples(self.X_train, self.Y_train)
            if self.alpha_range:
                model_init = {k:v for k,v in dtree_dict.items() if k in ['max_depth', 'min_node_samples', 'verbose_train', 'verbose_test']}
                model_train = {k:v for k,v in dtree_dict.items() if k in ['modality']}
                alpha = np.random.uniform(low = alpha_range[0], high = alpha_range[1])
                model = DecisionTree(**model_init)
                model.train(X_bootstrap, Y_bootstrap, alpha = alpha, **model_train)
            else:
                model_init = {k:v for k,v in dtree_dict.items() if k in ['max_depth', 'min_node_samples', 'verbose_train', 'verbose_test']}
                model_train = {k:v for k,v in dtree_dict.items() if k in ['alpha', 'modality']} # instead of drawing alpha, could use a random alpha value for different models in the ensemble, drawn randomly
                model = DecisionTree(**model_init)
                model.train(X_bootstrap, Y_bootstrap, **model_train)

            self.models.append(model)

    def test(self, X_test, Y_test):
        self.X_test = X_test
        self.Y_test = Y_test
      
        self._preds = self._get_model_pred()
        self.accuracy = self._accuracy(Y_test, self._preds)

        if self.verbose_test:
            print(f"\nFinal Ensemble Accuracy: {self.accuracy}%")

    def _bootstrap_samples(self, X, Y):
        bootstrap_idx = np.random.randint(low = 0, high = Y.size, size = (X.shape[0]))
        X_bootstrap = X[bootstrap_idx]
        Y_bootstrap = Y[bootstrap_idx]
        return X_bootstrap, Y_bootstrap
           
    def _get_model_pred(self):

        all_preds = [ ]

        for model in self.models:
            preds = model.test(self.X_test, self.Y_test)
            all_preds.append(preds)

        all_preds = np.array(all_preds)

        pred = np.apply_along_axis(self._most_common, axis = 0, arr = all_preds)
        self._preds.append(pred)
        return np.array(self._preds)

    def _most_common(self, all_preds):
        labels, freqs = np.unique(all_preds, return_counts = True)
        most_common_idx = np.argmax(freqs)
        return labels[most_common_idx]

    def _accuracy(self, Y, preds):
        acc = np.sum(Y.flatten() == preds.flatten()) / Y.size * 100
        return acc

if __name__ == "__main__":
    
    data = csv_to_numpy('data/DesTreeData.csv')
    train, test = train_test_split(data, train_split=.8)
    X_train, Y_train = x_y_split(train, y_col = 'last')
    X_test, Y_test = x_y_split(test, y_col = 'last')
  
    verbose_test = True
    n_bootstrap = 100
    dtree_dict = {
        'max_depth': 1000,
        'min_node_samples': 2,
        'alpha': 0,
        'verbose_train': False,
        'verbose_test': True,
        'modality': 'gini'
    }

    model = BaggedTrees(verbose_test = verbose_test) 
    model.train(X_train, Y_train, n_bootstrap = n_bootstrap, dtree_dict = dtree_dict)
    model.test(X_test, Y_test)

    '''model = DecisionTree(max_depth = 1000, min_node_samples=2, verbose = True)
    model.train(X_train, Y_train, alpha = .00001, modality = 'gini')
    model.test(X_test, Y_test)'''
