from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from models.classifier import Classifier
import numpy as np
import os
import pdb

class KNNClassifier_withPCA(Classifier):
    def __init__(self, dset, **kwargs):
        self.model_name = 'KNN-PCA'
        self.dset = dset
        self.k = kwargs.get('k', 5)
        self.weights = kwargs.get('weights', 'uniform')
        self.max_samples = kwargs.get('max_samples', None)
        self.clf = KNeighborsClassifier(n_neighbors=self.k, weights=self.weights)
        self.X = None
        self.y = None  
        self._load_data()
        self.lenEachX=self.X.shape[1]
        if self.max_samples is not None:
            self.X, self.y = self.X[0:self.max_samples, :], self.y[0:self.max_samples, :]
        self.do_pca=kwargs.get('do_PCA', True)
        self.pca_ncomp=kwargs.get('pca_ncomp', 200)
        if self.do_pca:
            self.pca = PCA(n_components=self.pca_ncomp)
            self.pca.fit(self.X)
            
    def train(self):
        if self.do_pca ==True:
            self.X = self._transform(self.X)
        
        self.clf.fit(self.X, self.y[:, 0])

    def test(self, X):
        if len(X.shape) > 2:
            X = X.reshape(-1, self.lenEachX)
        X = X.reshape(-1, self.lenEachX)
        if self.do_pca:
            X=self._transform(X)
        y_pred = self.clf.predict(X)
        return y_pred
    
    def _transform(self,data):
        return self.pca.transform(data)