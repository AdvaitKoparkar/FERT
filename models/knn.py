from sklearn.neighbors import KNeighborsClassifier
from models.classifier import Classifier
import numpy as np
import os

class KNNClassifier(Classifier):
    def __init__(self, dset, **kwargs):
        self.model_name = 'KNNClassifier'
        self.dset = dset
        self.k = kwargs.get('k', 5)
        self.weights = kwargs.get('weights', 'uniform')
        self.max_samples = kwargs.get('max_samples', None)
        self.clf = KNeighborsClassifier(n_neighbors=self.k, weights=self.weights)
        self.X = None
        self.y = None
        self._load_data()
        if self.max_samples is not None:
            self.X, self.y = self.X[0:self.max_samples, :], self.y[0:self.max_samples, :]

    def train(self):
        self.clf.fit(self.X, self.y[:, 0])

    def test(self, X):
        if len(X.shape) > 2:
            X = X.reshape(-1, self.X.shape[1])
        X = X.reshape(-1, self.X.shape[1])
        y_pred = self.clf.predict(X)
        return y_pred
