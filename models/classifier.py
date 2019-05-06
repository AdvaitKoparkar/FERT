from abc import ABC, abstractmethod
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

class Classifier(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def test(self):
        raise NotImplementedError

    def score(self, X=None, y=None, dloader=None):
        if dloader is None:
            y_pred = self.test(X)
            return np.sum(y_pred == y) / np.size(y)
        else:
            y_pred, acc = self.test(dloader=dloader)
            return acc

    def eval(self, X=None, dloader=None):
        if dloader is None:
            y_pred = self.test(X)
            return y_pred
        else:
            y_pred, acc = self.test(dloader=dloader)
            return y_pred

    def embeddings(self, dloader):
        embs = self.test(dloader=dloader, get_embeddings=True)
        return embs

    def predict(self, X, nn):
        if nn is False:
            X = np.reshape(X, (1,48*48))
            return self.eval(X=X)
        else:
            X = np.reshape(X, (1,1,self.size[0], self.size[1]))
            dl = torch.utils.data.TensorDataset(torch.tensor(X).type(torch.FloatTensor), torch.tensor(np.ones((1,1))).type(torch.FloatTensor))
            return self.eval(dloader=dl)

    def _load_data(self):
        if not (os.path.exists(os.path.join(self.dset.root, 'X_train.npy') or os.path.exists(os.path.join(self.dset.root, 'y_train.npy')))):
            print("Converting Train data to numpy")
            self.X, self.y = self._gen_numpy_data(self.dset)
            np.save(os.path.join(self.dset.root, 'X_train.npy'), self.X)
            np.save(os.path.join(self.dset.root, 'y_train.npy'), self.y)
        else:
            self.X = np.load(os.path.join(self.dset.root, 'X_train.npy'))
            self.y = np.load(os.path.join(self.dset.root, 'y_train.npy'))

    def _gen_numpy_data(self, dset):
        # Creating data in numpy format
        X_np, y_np = None, None
        for i, data in enumerate(tqdm(dset)):
            img, label = data
            if X_np is None:
                X_np = img.flatten()
                y_np = np.array([label])
            else:
                X_np = np.vstack((X_np, img.flatten()))
                y_np = np.vstack((y_np, [label]))


        return X_np, y_np
