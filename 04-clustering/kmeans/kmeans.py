import jax.numpy as jp
import jax.random as rdn
from termcolor import colored

class KMeans:

    def __init__(self, verbose = False, seed = 0):
        self.verbose = verbose
        self.seed = seed
        self.__key = rdn.key(seed = self.seed)

    def cluster(self, X, n_clusters, kpp = False, max_iter = 25):

        '''
        Cluster your dataset, `X`

        :param X: The unlabeled dataset.
        :type X: jax.numpy.ndarray
        :param k: The k total clusters.
        :type k: int
        :param kpp: If True, use kmeans++. If False, default to random centroid initialization
        :type kpp: bool

        '''

        self.X = X
        self.n_clusters = n_clusters
        self.kpp = kpp
        self.max_iter = max_iter
        self.centroids = None

        self._init_centroid()

        if isinstance(self.max_iter, int):
            for self.i in range(self.max_iter):
                converged = self._fit()
                if converged:
                    return
        elif self.max_iter == 'auto':
            self.i = 0
            while True:
                converged = self._fit()
                if converged:
                    return
                self.i += 1

    def _init_centroid(self):
        '''
        Init. Centroids
        '''
        self.centroids = rdn.choice(key = self._rdn_key(), a = X, shape = (self.n_clusters, ), replace = False, axis = 0)
        distances = [] 
        cent_dict = {}

        for i, centroid in enumerate(self.centroids):
            distance = jp.linalg.norm(self.X - centroid, ord = 2, axis = 1) # Distances are 1-dimensional of shape: (samples, ) -- as we're comparing the distance of each sample to the given centroid
            distances.append(distance)
        
        # The array denoting which centroid a sample belongs to. Each value in the array is the index of the centroid, each index of the value is the given index of the original sample, X.
        cent_idxs = jp.argmin(jp.array(distances), axis = 0)
       
        for idx, sample in zip(cent_idxs, self.X):
            idx = int(idx) 
            if idx not in cent_dict:
                cent_dict[idx] = []
            cent_dict[idx].append(sample)

        self.cent_dict = cent_dict
        print(f"Iteration: {1} | Loss: {self._wcss()}")

    def _fit(self): 
        if self.max_iter == 'auto':
            for i in self.cent_dict.keys():
                centroid = jp.mean(jp.array(self.cent_dict[i]), axis = 0)
                self.centroids = self.centroids.at[i].set(centroid)
            loss = self._wcss()
            print(f"Iteration: {self.i+2} | Loss: {self._wcss()}")
            if self._eval_stop(loss):
                return True

        else:
            for i in self.cent_dict.keys():
                centroid = jp.mean(jp.array(self.cent_dict[i]), axis = 0)
                self.centroids = self.centroids.at[i].set(centroid)

            loss = self._wcss()
            print(f"Iteration: {self.i + 2} | Loss: {self._wcss()}")
            if self._eval_stop(loss):
                return True

    def _wcss(self):
        '''
        Loss Func. WCSS.
        '''
        distances = []
        for i, centroid in enumerate(self.centroids):
            sample = jp.array(self.cent_dict[i])
            distances.append(jp.sum(jp.square(jp.linalg.norm(sample - centroid, ord = 2, axis = 1))))

        loss = jp.sum(jp.array(distances))
        return loss

    def _eval_stop(self, loss):

        n = 5

        try:
            self._loss_arr.append(int(loss))
        except:
            self._loss_arr = []
            self._loss_arr.append(int(loss))

        if len(self._loss_arr) >= 5 and len(set(self._loss_arr[-n:])) == 1:
            print(f'{colored("\nCONVERGED", "green", attrs = ['bold'])} | Final Loss: {self._wcss()}')
            return True


    def _rdn_key(self):
        return rdn.split(self.__key, num = 1)[0]
        

if __name__ == '__main__':

    X = jp.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6],
    [9.0, 11.0],
    [8.0, 2.0],
    [10.0, 2.0],
    [9.0, 3.0]
    ])

    verbose = True
    seed = 1
    n_clusters = 3
    max_iter = 10

    model = KMeans(verbose = verbose, seed = seed)
    model.cluster(X = X, n_clusters = n_clusters, max_iter = max_iter)
