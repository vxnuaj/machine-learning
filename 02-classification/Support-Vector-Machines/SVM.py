from nue.preprocessing import x_y_split, csv_to_numpy
import numpy as np

'''

TODO

Linear SVMs Hard Margin
- [X] Train and test the SVM on a non-complex dataset (traindata.csv)
- [X] Test the SVM (implement multivariable / multifeature) on a more complex (4 + features) dataset.
- [X] Add to Nue.

Linear SVMs Soft Margin
- [X] Train and test the SVM on a non-complex dataset (traindata.csv)
- [X] Test the SVM on a more complex (4 + features) dataset.
- [X] Add to Nue.

Dual SVMs
- [X] Theory
- [] SkLearn
'''
import numpy as np
from nue.metrics import svm_hinge_loss
from nue.preprocessing import x_y_split

class SVM():
    def __init__(self, seed:int = None, modality = 'soft', C = .01):
       
        '''
        :param seed: Set the random seed for initializing parameters. Based on numpy.random.default_rng() 
        :type seed: int
        
        :param modality: Set the modality for the SVM. Choose between soft and hard, for soft-margin and hard-margin respectively. 
        :type modality: str 
        
        :param C: Set the regularization strength when using modality as 'soft'
        :type C: float
        '''

        self.seed = seed
        self.modality = modality 
        self.C = C
         
        self.X_train = np.empty(0)
        self.Y_train = np.empty(0)
        self.alpha = .0001
        self.epochs = 250

        self.__num_features = None
        self.__params = []
        self.__gradients = []

    def _init_params(self):
        """
        Initialize the parameters (weights and bias) for the SVm, based on the chosen seed in the init method.
        
        :return: Tuple contianing the weights (w) and bias (b) 
        :rtype: tuple
        """     
        if self.seed == None:
            w  = np.random.rand(1, self.__num_features)
            b = np.zeros((1,1))
            self.__params = [w, b] 
        else:
            rng = np.random.default_rng(seed = self.seed)
            w = rng.normal(size = (1, self.__num_features))
            b = np.zeros((1, 1))
            self.__params = [w, b] 
        
        return self.__params

    def _forward(self):
        '''
        Compute the forward pass to compute the predicted probabilities
        
        :return: The predicted probabilities, or the functional margin
        :rtype: numpy.ndarray 
        '''
        w, b = self.__params

        self._output = np.dot(w, self.X_train) + b
        return self._output

    def accuracy(self, y):
        pred = np.sign(self._output) 
        self.train_acc = np.sum(y == pred) / y.size * 100
        return self.train_acc
    
    def _backward(self):
        '''
        Compute the gradients for the parameters (w and b) with a backward pass.

        :return: List containing the gradients of the weights (dw) and bias (db)
        :rtype: list
        ''' 
       
        if self.modality == 'soft':  
            dz = self.C * np.maximum(0, -self.Y_train) 
            dw = np.dot(dz, self.X_train.T) / self.Y_train.size
            db = np.sum(dz) / self.Y_train.size
            self.__gradients = [dw, db] 
        else:
            dz = np.maximum(0, -self.Y_train) 
            dw = np.dot(dz, self.X_train.T) / self.Y_train.size
            db = np.sum(dz) / self.Y_train.size
            self.__gradients = [dw, db] 
        return self.__gradients
    
    def _update(self):
        '''
        Update the weights and bias using gradient descent.
        
        :return: list contianing the update weights (w) and bias (b) 
        :rtype: list 
        '''

        dw, db = self.__gradients
        w, b = self.__params
        
        w -= self.alpha * dw
        b -= self.alpha * db
        
        self.__params = [w, b]

        return self.__params
    
    def support_vector(self):
        '''
        Identify the support vectors of the given SVM based on their functional margin and comute their geometric margin.
        '''

        k = 6
        w, _ = self.__params 

        self._functional_margin = (self.Y_train * self._output).flatten()
        min_indices = np.argpartition(self._functional_margin, k)[:k]
        self.support_vectors = self.X_train[:, min_indices]
        weight_norm = np.linalg.norm(w, ord = 2, axis = 1)
        if weight_norm == 0:
            raise ValueError("The norm of the weight vector is zero, cannot compute geometric margin.")

        self.geometric_margins = self._functional_margin[min_indices] / weight_norm
        print(self.geometric_margins.shape)
        return self.support_vectors, self.geometric_margins
        
    def _gradient_descent(self, verbose:bool, metric_freq:int):
        '''
        Perform gradient descent to train the logistic regression model
        
        :return: List containing the final weights (w) and bias (b).
        :rtype: List
        '''

        print(f"Model Training!")
        
        for epoch in range(self.epochs):
            self._output = self._forward()
            
            self.train_loss = svm_hinge_loss(self._output, self.Y_train, self.__params, self.modality, self.C)
            self.train_acc = self.accuracy(self.Y_train)

            self.__gradients = self._backward()
            self.__params = self._update()
            
            if verbose == True: 
                if epoch % metric_freq == 0:
                    print(f"Epoch: {epoch}")
                    print(f"Loss {self.train_loss}")
                    print(f"Accuracy: {self.train_acc}%\n")
                
        if verbose == True:
            print(f"Final Training Loss: {self.train_loss}")
            print(f"Final Training Accuracy: {self.train_acc}%\n")

        self.support_vectors = self.support_vector()

        return self.__params

    def train(self, X_train:np.ndarray, Y_train:np.ndarray, alpha:float = .0001, epochs:int = 250, verbose:bool = False, metric_freq:int = 1 ):
        '''
        Train the SVM
         
        :param X_train: The input data of shape (features, samples)
        :type X_train: numpy.ndarray
        :param Y_train: The target labels of shape (1, samples). If binary, labels must be -1 or 1.
        :type Y_train: numpy.ndarray
        :param alpha: The learning rate for gradient descent
        :type alpha: float 
        :param epochs: The number of epochs for training
        :type epochs: int

        :param verbose: If True, will print out training progress of the model
        :type verbose: bool
        :param metric_freq: Will not apply if verbose is set to False. 
      
            Will print out epoch and loss at the epoch frequency set by metric_freq 
        
        :type metric_freq: int

        :return: list containing the final weights (w) and bias (b).
        :rtype: list
        '''
       
        self.X_train = X_train
        self.Y_train = Y_train
        self.alpha = alpha
        self.epochs = epochs 
        
        self.__num_features = X_train.shape[0]
        
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be type bool!")
        
        self.__params = self._init_params()
        self.__params = self._gradient_descent(verbose, metric_freq) 

        return self.__params

    def test(self, X_test:np.ndarray, Y_test:np.ndarray, verbose:bool = False ):
        '''
        Test the SVM
       
        :param X_test: The validation features, shape (features, samples).
        :type X_test: numpy.ndarray
        :param Y_test: The validation labels, shape (1, samples). If binary, labels must be -1 or 1.
        :type Y_test: numpy.ndarray
        :param verbose: If true, will print out loss and r2 score post-test.
        :type verbose: bool  
        '''        

        self.X_test = X_test
        self.Y_test = Y_test

        if not isinstance(verbose, bool):
            raise ValueError("verbose must be type bool!")

        print("Model testing!")
        
        w, b = self.__params
        self._output = np.dot(w, X_test) + b
       
        self.test_loss = svm_hinge_loss(self._output, self.Y_test, self.__params, modality = 'hard')
        self.test_acc = self.accuracy(Y_test)
        
        print(f"Model tested!\n") 
        
        if verbose:
            print(f"Final test loss: {self.test_loss}")
            print(f"Final test accuracy: {self.test_acc}%\n")
        
        return self.test_loss, self.test_acc        

    def inference(self, X_inf:np.ndarray, Y_inf:np.ndarray, verbose:bool = False):
        '''
        Inference of the SVM
        
        :param X_inf: The validation features, shape (featuers, samples)
        :type X_test: numpy.ndarray
        :param Y_test: The validation labels, shape (1, samples). If binary, labels must be -1 or 1.
        :type Y_test: numpy.ndarray
        :param verbose: If true, will print out loss and r2 score post-test.
        :type verbose: bool  
        ''' 
        
        self.X_inf = X_inf
        self.Y_inf = Y_inf
        
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be type bool!")
        
        w, b = self.__params
        self._output = np.dot(w, X_inf) + b
        
        self.inf_loss = svm_hinge_loss(self._output, Y_inf, self.__params, modality = 'hard')
        self.inf_acc = self.accuracy(self.Y_inf)

        self.pred = np.maximum(self._output, 0)        

        return self.pred

    def metrics(self, mode = 'train'):
        
        """
       
        Prints the training and / or testing accuracy and loss for the model, dependent on `mode`
        
        :param mode: The metrics to be printed, defined by 'train', 'test', or 'both'
        :type mode: str
        
        """ 
        
        if mode not in ['train', 'test', 'both']:
            raise ValueError("mode must be type str, of 'both', 'train', or 'test'!") 
        
        if mode in ['both', 'test']: 
            if self.train_loss and not hasattr(self, 'test_loss'):
                raise ValueError("You haven't tested the model yet!")
            elif not hasattr(self, 'train_loss'):
                raise ValueError("You haven't trained the model yet!") 
  
        if mode == 'both':
            print(f"Train loss: {self.train_loss} | Train accuracy: {self.train_acc}%\nTest loss: {self.test_loss} | Test accuracy: {self.test_acc}%")
        elif mode == 'test':
                print(f"Test loss: {self.test_loss} | Test accuracy: {self.test_acc}%") 
      
        elif mode == 'train':
            if not hasattr(self, 'train_loss'):
                raise ValueError("You haven't trained the model yet!")
            print(f"Train loss: {self.train_loss} | Train accuracy: {self.train_acc}%")  

    @property
    def X_train(self):
        return self._X_train 
    
    @X_train.setter
    def X_train(self, X_train):
        if not isinstance(X_train, np.ndarray):
            raise ValueError("X_train must be type numpy.ndarray!")
        self._X_train = X_train
    
    @property
    def Y_train(self):
        return self._Y_train  
       
    @Y_train.setter
    def Y_train(self, Y_train):
        if not isinstance(Y_train, np.ndarray):
            raise ValueError("Y_train must be type numpy.ndarray!") 
        self._Y_train = Y_train
   
    @property
    def X_test(self):
        return self._X_test
   
    
    @X_test.setter
    def X_test(self, X_test):
        if not isinstance(X_test, np.ndarray):
            raise ValueError("X_train must be type numpy.ndarray!")
        self._X_test = X_test
    
    @property
    def Y_test(self):
        return self._Y_test 
       
    @Y_test.setter
    def Y_test(self, Y_test):
        if not isinstance(Y_test, np.ndarray):
            raise ValueError("Y_train must be type numpy.ndarray!") 
        self._Y_test = Y_test
   
    @property
    def X_inf(self):
        return self._X_inf
    
    @X_inf.setter
    def X_inf(self, X_inf):
        if not isinstance(X_inf, np.ndarray):
            raise ValueError("X_train must be type numpy.ndarray!")
        self._X_inf = X_inf
    
    @property
    def Y_inf(self):
        return self._Y_inf  
       
    @Y_inf.setter
    def Y_inf(self, Y_inf):
        if not isinstance(Y_inf, np.ndarray):
            raise ValueError("Y_train must be type numpy.ndarray!") 
        self._Y_inf = Y_inf
   
    @property
    def alpha(self):
        return self._alpha 
   
    @alpha.setter
    def alpha(self, alpha):
        if not isinstance(alpha, float):
            raise ValueError("alpha must be type float!")
        self._alpha = alpha     
    
    @property
    def epochs(self):
        return self._epochs
    
    @epochs.setter
    def epochs(self, epochs):
        if not isinstance(epochs, int):
            raise ValueError("epochs must be type int!")
        self._epochs = epochs

    @property
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, seed):
        if not isinstance(seed, int) and seed is not None:
            raise ValueError("seed must be type int or set as none!")
        self._seed = seed

    @property
    def modality(self):
        return self._modality
    
    @modality.setter
    def modality(self, modality):
        if modality.lower() not in ['soft', 'hard']: 
            raise ValueError("modality must be 'soft' or 'hard'!")
        self._modality = modality
        
    @property
    def C(self):
        return self._C
    
    @C.setter
    def C(self, C):
        if self.modality == 'hard':
            self._C = None
        else:
            self._C = C
    