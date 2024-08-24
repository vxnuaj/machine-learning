import numpy as np

class PCA:
    def __init__(self, n_components = None):
        self.X = None
        self.n_components = n_components
        self._eigvals = None
        self._eigvecs = None
        self._cov = None
        self.X = None
        self.X_reduced = None
    
    def fit_transform(self, X):
        #Each row of X is a sample 
        self.X = X
        self._X_mean = self.X - np.mean(self.X, axis = 0)
        self._cov = self._cov_matrix()
        
        self._eigvals, self._eigvecs = np.linalg.eig(self._cov.astype(float))
        
        eig_idxs = np.argsort(self._eigvals)[::-1]
        
        self._eigvals = self._eigvals[eig_idxs]
        self._eigvecs = self._eigvecs[:, eig_idxs]
        self._components = self._eigvecs[:, :self.n_components] 
       
        return np.dot(self._X_mean, self._components)
        
        return self.X_reduced 
        
    def _reduce_dims(self):
        return np.dot(self._X_mean, self._eigvecs)
         
    def _cov_matrix(self):
        #Compute Covariance. 
        #Make sure that it's along axis = 0, as we want to comptue the mean over the rows, since they correspond to the samples.
        #Or just use np.cov(), lol
        cov = np.dot((self.X - np.mean(self.X, axis = 0)).T, (self.X - np.mean(self.X, axis = 0))) / (self.X.shape[0] - 1) 
        return cov 

    @property
    def n_components(self):
        return self._n_components 
    
    @n_components.setter
    def n_components(self, n_components):
        if n_components and self.X:
            assert n_components < self.X.shape[1], 'n_components must be less than the number of features. you can only reduce dimensionality when using PCA.' 
        self._n_components = n_components
        
    