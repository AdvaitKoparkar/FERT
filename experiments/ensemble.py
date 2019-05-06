import numpy as np
from tqdm import tqdm
from dataloaders.fer_loader import FERDataset
from experiments.classification import Classification
from torch.utils.data import DataLoader
from scipy.stats import mode
import torch

class Ensemble(object):
    def __init__(self, classifiers, hyper_params, classifier_type, **kwargs):
        self.model_name = 'Ensemble'
        self.classifiers = classifiers
        self.hyper_params = hyper_params
        self.neural_net = classifier_type
        self.train_dset = FERDataset()
        self.model_weights = kwargs.get('model_weights', [None]*len(self.classifiers))
        self.classification_list = []
        for cidx, classifier in enumerate(self.classifiers):
            self.classification_list.append(Classification(**classifier))
        if self.hyper_params is not None:
            assert len(self.hyper_params) == len(self.classifiers)

    def grid_search(self, restore=False):
        self.models = []
        for cidx, classifier in enumerate(self.classifiers):
            classification = Classification(**classifier)
            if restore is False:
                print("Training %s: " %classification.classifier.__name__)
                classification.grid_search(self.hyper_params[cidx])
            else:
                if self.model_weights[cidx] is None:
                    print("Restoring %s" %classification.classifier.__name__)
                    classification.restore_classifier()
                else:
                    print("Restoring %s" %self.model_weights[cidx])
                    classification.restore_classifier(fpath=self.model_weights[cidx])
            self.models.append(classification)

    def gen_metrics(self):
        for model in self.models:
            model.gen_metrics()

    def score(self, X=None, y=None, dloader=None):
        y_pred = self.eval()
        y = self.models[0].get_data()[:, 0]
        return np.sum(y == y_pred) / y_pred.shape[0]

    def eval(self):
        y_pred = None
        for cidx, model in enumerate(self.models):
            y = model.eval()
            if y_pred is None:
                y_pred = y
            else:
                y_pred = np.hstack((y_pred, y))
        y_pred = mode(y_pred, axis=1)[0]
        return y_pred[:, 0]

    def single_eval(self, X):
        y_pred = []
        for cidx, classification in enumerate(self.classification_list):
            y_pred.append(classification.single_eval(X[:, :, 0:1])[0][0])
        return mode(y_pred)
