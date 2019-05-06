from abc import ABC, abstractmethod

class Cluster(object):
    def __init__(self):
        pass

    @abstractmethod
    def cluster(self, frames):
        raise NotImplementedError
