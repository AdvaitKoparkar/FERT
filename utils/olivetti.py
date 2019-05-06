import numpy as np
import os
import pickle
from sklearn.datasets import fetch_olivetti_faces
from models.dbscan import DBSCAN
from models.mean_shift import MeanShift
from scipy.misc import imresize
from sklearn import metrics
import pdb

class OlivettiDataset(object):
    def __init__(self, **kwargs):
        self.data_home = kwargs.get('data_home', "data/olivetti")
        self.data_file = kwargs.get('data_file', "olivetti.pkl")
        self.cluster = kwargs.get('cluster', MeanShift())

    def generate_dataset(self):
        dataset = fetch_olivetti_faces(data_home="data/olivetti", shuffle=True)
        images = dataset.images
        targets = dataset.target
        images *= 255
        images = images.astype(np.uint8)
        X_olivetti = []
        y_olivetti = []
        for idx, image in enumerate(images):
            X_olivetti.append(imresize(image, (48,48)))
            y_olivetti.append(targets[idx])

        with open(os.path.join(self.data_home, self.data_file), "wb") as fh:
            pickle.dump({'X': X_olivetti, 'y': y_olivetti}, fh)


    def generate_results(self):
        with open(os.path.join(self.data_home, self.data_file), "rb") as fh:
            olidict = pickle.load(fh)
        X_olivetti = olidict['X']
        y_olivetti = olidict['y']
        y_pred, _ = self.cluster.cluster(X_olivetti)
        with open(os.path.join(self.data_home, "y_pred.pkl"), "wb") as fh:
            pickle.dump(y_pred, fh)
        print("Adjusted rand Score:",metrics.adjusted_rand_score(y_olivetti, y_pred))
        print("AMI",metrics.adjusted_mutual_info_score(y_olivetti, y_pred))
        print("V_measure", metrics.v_measure_score(y_olivetti, y_pred))
        print("FMS",metrics.fowlkes_mallows_score(y_olivetti, y_pred))
#        pdb.set_trace()
