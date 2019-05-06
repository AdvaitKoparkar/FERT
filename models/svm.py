from sklearn.svm import SVC
from models.classifier import Classifier
import numpy as np
import os

class SVMClassifier(Classifier):
    def __init__(self, dset, **kwargs):
        self.model_name = 'SVMClassifier'
        self.dset = dset
        self.C = kwargs.get('C', 1)
        self.kernel = kwargs.get('kernel', 'rbf')
        self.decision_function_shape = kwargs.get('decision_function_shape', 'ovo')
        self.max_samples = kwargs.get('max_samples', None)
        self.clf = SVC(C = self.C, kernel = self.kernel, decision_function_shape = self.decision_function_shape, gamma = 'auto')
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
