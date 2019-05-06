from abc import ABC, abstractmethod

class Embeddings(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_embeddings(self, face_list):
        raise NotImplementedError
