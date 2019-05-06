from models.cluster import Cluster
from utils.nn_embeddings import NNEmbeddings
import sklearn.cluster
import numpy as np

class DBSCAN(Cluster):
    def __init__(self, **kwargs):
        self.metric = kwargs.get('metric', 'cosine')
        self.eps = kwargs.get('eps', 0.05)
        self.min_samples = kwargs.get('min_samples', 1)
        self.n_jobs = kwargs.get('n_jobs', 1)
        self.encoder = kwargs.get('encoder', NNEmbeddings())
        self.clt = sklearn.cluster.DBSCAN(eps=self.eps,
                                          min_samples = self.min_samples,
                                          n_jobs=self.n_jobs,
                                          metric=self.metric)

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
