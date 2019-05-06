import numpy as np
from sklearn.linear_model import LogisticRegression
from models.classifier import Classifier
import os

class LogRegClassifier(Classifier):
    def __init__(self, dset, **kwargs):
        self.model_name='LogRegClassifier'
        self.dset= dset
        self.multi_class=kwargs.get('multi_class','multinomial')
        self.max_samples = kwargs.get('max_samples', 1000)
        self.solver=kwargs.get('solver', 'newton-cg')
        self.C=kwargs.get('C',1)
        self.clf= LogisticRegression(solver=self.solver, multi_class=self.multi_class)
        self.X=None
        self.y=None
        self._load_data()
        if self.max_samples is not None:
            self.X, self.y =self.X[0:self.max_samples,:], self.y[0:self.max_samples,:]

    def train(self):
        self.clf.fit(self.X,self.y[:,0])

    def test(self, X):
        X=X.reshape(-1,self.X.shape[1])
        y_pred=self.clf.predict(X)
        return y_pred
