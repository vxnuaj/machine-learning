import numpy as np
from nue.metrics import dt_accuracy, entropy, gini

class Node():
    
    '''
    Initializes a Node of a Decision Tree. Primarily for internal use of the `DecisionTree` class. 
   
    :param value: The value of the node, if the Node is a leaf or pure node.
    :type value: float or int 
    :param left_node: The left node of the given Node instance. Recursively grown via the `DecisionTree._grow_tree()`
    :type left_node: Node 
    :param right_node: The right node of the given Node instance. Recursively grown via the `DecisionTree._grow_tree()`.
    :type right_node: Node
    :param feature: The optimal feature index for which to split the samples within the current Node
    :type feature: int
    :param threshold: The optimal threshold value, within the range of the optimal feature column vector, to split the current set of samples within the current Node, into left or right Nodes.
    :type threshold: float or int
    ''' 
    
    def __init__(self, value = None, left_node = None, right_node = None, feature = None, threshold = None):

        self.value = value
        self.left_node = left_node
        self.right_node = right_node
        self.feature = feature
        self.threshold = threshold

    def _is_leaf(self):
        
        '''
        Assess if the current Node is a leaf node or not. 
        
        :return: A boolean value, True if self.value isn't None. Otherwise, if it is None, returns False 
        :rtype: bool
        ''' 
        
        return self.value is not None # Returns True if self.value isn't None. Otherwise, if it is None, returns False.

    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, value):
        if not isinstance(value, (float, np.floating, int, np.integer, type(None))):
            raise ValueError('value must be type int or float.')
        self._value = value
   
    @property
    def left_node(self):
        return self._left_node
    
    @left_node.setter
    def left_node(self, left_node):
        if not isinstance(left_node, (Node, type(None))):
            raise ValueError('left_node must be an instance of Node.')
        self._left_node = left_node        
    
    @property
    def right_node(self):
        return self._right_node
    
    @right_node.setter
    def right_node(self, right_node):
        if not isinstance(right_node, (Node, type(None))):
            raise ValueError('right_node must be an instance of Node')
        self._right_node = right_node
        
    @property
    def feature(self):
        return self._feature
    
    @feature.setter 
    def feature(self, feature):
        if not isinstance(feature, (int, type(None))):
            raise ValueError('feature must be type int')
        self._feature = feature 
        
    @property
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold(self, threshold):
        if not isinstance(threshold, (float, np.floating, int, np.integer, type(None))):
            raise ValueError('threshold must be type float or int or None.')
        self._threshold = threshold
         
class DecisionTree():
    
    '''
    Initialize the DecisionTree. 
    
    :param max_depth: The maximum depth allowed in the decision tree.
    :type max_depth: int
    :param min_sample_split: The least amount of samples allowed for a Node to split
    :type min_sample_split: int
    :param modality: The modality for fitting the tree. Entropy is the default. Currently supports 'entropy' or 'gini' 
    ''' 
    
    def __init__(self, max_depth, min_sample_split, modality = 'entropy'):
        
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split 
        self.modality = modality

        self.n_leaf = 0
        self.root = None 

    def fit(self, X_train, Y_train, alpha = None, verbose = False):
       
        '''
        Fit the Decision Tree.
        
        :param X_train: The training data for the Decision Tree, of shape (samples, feature)
        :type X_train: numpy.ndarray
        :param Y_train: The labels for the corresponding X_train, of shape (samples, ) or (samples, 1)
        :type X_train: numpy.ndarray
        :param verbose: The verbosity for fitting the Decision Tree. If True, during training, expect a shit load of output.
        :type verbose: bool
        :param alpha: The cost complexity parameter, similar to regularization for lin, log, or nn
        '''
        
        self.X_train = X_train
        self.Y_train = Y_train
        self.alpha = alpha
        self.verbose_fit = verbose
        
        self.root = self._grow_tree(self.X_train, self.Y_train)
      
        
    def _grow_tree(self, X, Y, depth = 0):
      
        '''
        Grow the Decision Tree recursively.
        
        :param X: The training data, used to grow the decision tree, of shape (samples, features)
        :type X: numpy.ndarray
        :param Y: The labels for the corresponding X, of shape (samples,) or (samples, 1)
        :type Y: numpy.ndarray 
        :param depth: The current depth of the given Node in the Decision Tree
        :type: int
        
        :return: The root node, holding the left_node(s) and right_node(s) splits as Node.left_node and Node.right_node
        :return type: Node instance
        '''
 
        n_samples, n_feature = X.shape
        n_labels = len(np.unique(Y))
       
        # Stopping Criteria 
        if (depth > self.max_depth or n_labels == 1 or n_samples < self.min_sample_split):
            self.n_leaf += 1
            leaf_value = self._most_common_label(Y)
            return Node(value = leaf_value) 
         
        best_feat, best_thresh = self._best_split(X, Y)
       
        # If there is no best feature or best threshold index, return the Node. Occurs if the information_gain doesn't exceed -1 for the current node.
        # This then only occurs when we introduce the alpha parameter (aka complexity parameter), as a means of regularization
        # See here - https://www.youtube.com/watch?v=Tg2OGohaUTc
        # The alpha parameter adds a penalty to the weighted entropy, thereby when we compute the information entropy as parent_entropy - weighted_entropy, 
        # some of the values for information entropy are negative. The amount of negative values are determiend by the magnitude of alpha.
        # Then, given that, more weights are pruned with no optimal solution the bigger alpha (complexity parameter), reducing overfitting.
        
        if best_feat is None or best_thresh is None:
            leaf_value = self._most_common_label(Y)
            return Node(value = leaf_value) 
        
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh) 
     
        depth += 1
        
        if self.verbose_fit:
            print(f"Tree Depth: {depth}") 
      
        left_node = self._grow_tree(X[left_idxs, :], Y[left_idxs], depth) 
        right_node = self._grow_tree(X[right_idxs, :], Y[right_idxs], depth) 
     
        return Node(left_node = left_node, right_node = right_node, feature = best_feat, threshold = best_thresh) 
        
    def _best_split(self, X, Y):
       
        '''
        Identify the best splits of the node, feature and threshold split, based on the information gain or gini index. 
       
        :param X: The  training data, used to grow the decision tree, of shape (samples, features)
        :type X: numpy.ndarray
        :param Y: The labels for the corresponding X, of shape (samples,) or (samples, 1)
        :type Y: numpy.ndarray 
        
        :return best_feat: The best feature index of X, as a splitting feature index
        :rtype best_feat: int
        :return best_thresh: The best threshold to split the samples X, with the correpsonding feature index.
        :rtype best_thresh: int or float
        '''
        
        n_samples, n_features = X.shape 
        best_gain = -1 
        best_feat = None
        best_thresh = None
         
        for feat in range(n_features):
            X_col = X[:, feat]
            thresholds = np.unique(X[:, feat])

            for thresh in thresholds:
                self.information_gain = self._information_gain(X_col, Y, thresh)

                if self.information_gain > best_gain:
                    best_gain = self.information_gain
                    best_feat = feat
                    best_thresh = thresh
       
        return best_feat, best_thresh

    def _split(self, X_col, thresh):
        
        '''
        Split the optimal feature column of X (X_col) given the threshold (`thresh`)
        
        :param X_col: The feature column of X to split using `thresh`
        :type X_col: numpy.ndarray
        :param thresh: The given threshold value to split X_col
        :type thresh: float or int
        
        :return left_idxs: The sample indices corresponding to samples that will be split to the left_node
        :type left_idxs: int
        :return right_idxs: The sample indices corresponding to samples that will be split to the right_node
        :type right_idxs: int
        ''' 
        
        left_idxs = np.argwhere(X_col < thresh).flatten() # what happens if we don't flatten either?
        right_idxs = np.argwhere(X_col >= thresh).flatten()
        return left_idxs, right_idxs
                
    def _information_gain(self, X_col, Y, thresh):
        
        '''
        Compute the information gain for a given split.
        
        :param X_col: The feature column of X.
        :type X_col: numpy.ndarray 
        :param Y: The set of labels corresponding to samples of X
        :type Y: numpy.ndarray
        :param thresh: The given threshold to split `X_col`
        
        :return: The information gain of the given split
        :rtype information_gain: float or int
        ''' 
        
        left_idxs, right_idxs = self._split(X_col, thresh)  
        
        n = len(Y)
        n_l = len(left_idxs)
        n_r = len(right_idxs)  
    
        if self.modality == 'entropy': 
           
            parent_entropy = entropy(Y) 
            left_entropy, right_entropy = entropy(Y[left_idxs]), entropy(Y[right_idxs]) 
            
            if self.alpha: 
                weighted_entropy = ((n_l / n) * left_entropy + (n_r / n) * right_entropy) + self.alpha * np.abs(self.n_leaf)
            else:
                weighted_entropy = (n_l / n) * left_entropy + (n_r / n) * right_entropy
 
            information_gain = parent_entropy - weighted_entropy 
         
        elif self.modality == 'gini':
            parent_gini = gini(Y)
            left_gini, right_gini = gini(Y[left_idxs]), gini(Y[right_idxs]) 
            weighted_gini = (n_l / n) * left_gini + (n_r / n) * right_gini
            information_gain = parent_gini - weighted_gini     
             
        return information_gain
        
    def _most_common_label(self, Y):
        
        '''
        Compute the most common label in a given node, through the given labels (Y) within that node.
         
        :param Y: The labels for a given node
        :type Y: numpy.ndarray 
        
        :return: The most common label in Y
        :rtype: float or int
        ''' 
        
        unique_labels, counts = np.unique(Y.flatten(), return_counts = True)
        most_common_index = np.argmax(counts)
        return unique_labels[most_common_index]


    def predict(self, X_test, Y_test = None, verbose = False):
        
        '''
        Predict a label given a set of testing samples and labels
        
        :param X_test: The set of testing samples of shape (samples, features)
        :type X_test: numpy.ndarray 
        :param Y_test: The set of labels correpsonding to X_test of shape (samples, 1) or (samples, )
        :type Y_test: numpy.ndarray
        
        :return: The predictions of the decision tree, for x in X_test
        :rtype: numpy.ndarray
        ''' 
        
        self.X_test = X_test 
        self.Y_test = Y_test
        self.verbose_predict = verbose

        pred = np.array([self._traverse(x) for x in X_test])

        self.test_acc = dt_accuracy(Y_test.flatten(), pred) 
       
        if self.verbose_predict:
            self.metrics()
            
        return pred

    def _traverse(self, x):
        
        '''
        Traverse the Decision Tree
        
        :param x: A single sample (column) of X_test
        :type x: numpy.ndarray
        
        :return: The predicted value, based on a leaf node of the Decision Tree, for a given sample x
        :rtype x: float or int
        '''
        
        node = self.root
   
        while not node._is_leaf(): 
        
            if x[node.feature] < node.threshold:
                node = node.left_node
            elif x[node.feature] >= node.threshold:
                node = node.right_node
        
        return node.value
       
    def metrics(self):
        print(f"\nTotal Leaf Nodes: {self.n_leaf}") 
        print(f"Accuracy: {self.test_acc}%")
       
    @property
    def X_train(self):
        return self._X_train
    
    @X_train.setter
    def X_train(self, X_train):
        if not isinstance(X_train, (np.ndarray)):
            raise ValueError('X_train must be type numpy.ndarray.')
        self._X_train = X_train
   
    @property
    def Y_train(self):
        return self._Y_train
    
    @Y_train.setter
    def Y_train(self, Y_train):
        if not isinstance(Y_train, (np.ndarray)):
            raise ValueError('Y_train must be type numpy.ndarray.')   
        self._Y_train = Y_train 
      
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, alpha):
        if not isinstance(alpha, (int, float, type(None))):
            raise ValueError('alpha must be type float.')
        self._alpha = alpha 
       
    @property
    def verbose_fit(self):
        return self._verbose_fit
    
    @verbose_fit.setter
    def verbose_fit(self, verbose_fit):
        if not isinstance(verbose_fit, bool):
            raise ValueError('verbose_fit must be type bool.')
        self._verbose_fit = verbose_fit

    @property
    def modality(self):
        return self._modality
    
    @modality.setter
    def modality(self, modality):
        if modality.lower() not in ['entropy', 'gini']:
            raise ValueError("modality must be 'entropy or 'gini'.")
        self._modality = modality
        
    @property
    def X_test(self):
        return self._X_test
    
    @X_test.setter
    def X_test(self, X_test):
        if not isinstance(X_test, np.ndarray):
            raise ValueError('X_test must be type numpy.ndarray.')
        self._X_test = X_test
        
    @property
    def Y_test(self):
        return self._Y_test
    
    @Y_test.setter
    def Y_test(self, Y_test):
        if not isinstance(Y_test, np.ndarray):
            raise ValueError('Y_test must be type numpy.ndarray.')
        self._Y_test = Y_test 
       
    @property
    def verbose_predict(self):
        return self._verbose_predict
   
    @verbose_predict.setter
    def verbose_predict(self, verbose_predict):
        if not isinstance(verbose_predict, bool):
            raise ValueError('verbose must be type bool')
        self._verbose_predict = verbose_predict 