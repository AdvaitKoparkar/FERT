from models.cluster import Cluster
from utils.nn_embeddings import NNEmbeddings
import sklearn.cluster
import numpy as np

class MeanShift(Cluster):
    def __init__(self, **kwargs):
        self.bandwidth = kwargs.get('bandwidth', 100)
        self.encoder = kwargs.get('encoder', NNEmbeddings())
        self.clt = sklearn.cluster.AffinityPropagation()

    def cluster(self, frames, locations=None):
        hashes = []
        encodings = []
        if locations is None:
            locations = [[(0, 47, 47, 0)]]*len(frames)
        for frame, loc in zip(frames, locations):
            hashes.append(hash(frame.tostring()))
            encodings += self.encoder.get_embeddings(frame, loc)
        encodings = np.asarray(encodings)
        self.clt.fit(encodings)
        labels = self.clt.labels_
        return labels, hashes
