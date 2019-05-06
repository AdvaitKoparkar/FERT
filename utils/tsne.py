import sklearn.decomposition
import sklearn.manifold
import matplotlib.pyplot as plt
import os
import numpy as np
from models.simple_cnn import SimpleCNN
from dataloaders.fer_loader import FERDataset
from experiments.classification import Classification
from copy import copy
import pickle
import seaborn as sns
import pdb

class TSNE(object):
    def __init__(self, **kwargs):
        self.mode = kwargs.get('mode', 'train')
        self.dset_name = kwargs.get('dset', None)
        self.dset_root = kwargs.get('dset_root', './data/fer2013')
        self.dset_filename = kwargs.get('dset_filename', 'fer2013.csv')
        self.perplexity = kwargs.get('perplexity', 50.0)
        self.n_components_tsne = kwargs.get('n_components_tsne', 2)
        self.learning_rate = kwargs.get('learning_rate', 200.0)
        self.metric = kwargs.get('metrics', 'euclidean')
        self.n_components_pca = kwargs.get('n_components_pca', 7)
        self.save_path = kwargs.get('save_path', os.path.join("data", "fer2013", "data_tsne_train_resnet.pkl"))
        self.model_save_path = kwargs.get('model_save_path', None)
        self.model = kwargs.get('model', None)
        self.train_dset = self.dset_name(root=self.dset_root, filename=self.dset_filename, mode='train', tensor=False)
        self.val_dset = self.dset_name(root=self.dset_root, filename=self.dset_filename, mode='val', tensor=False)
        self.test_dset = self.dset_name(root=self.dset_root, filename=self.dset_filename, mode='test', tensor=False)
        self.classes = kwargs.get('classes', {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'})
        self._load_data()
        if not os.path.exists(self.save_path):
            indices = np.arange(self.X_train.shape[0])
            np.random.shuffle(indices)
            if self.model is None:
                self.pca = sklearn.decomposition.PCA(n_components=self.n_components_pca)
                self.X = self.pca.fit_transform(self.X_test)
            else:
                self.classification = Classification(**{'classifier': self.model, 'dset':FERDataset, 'neural_net':True, 'save_path': self.model_save_path})
                self.X = self.classification.embeddings()
            self.X = self.X[indices[0:3500], :]
            self.y = self.y_train[indices[0:3500], :]
        # if self.mode == 'train':
        #     self.X = self.X_train[indices[0:1000], :]
        #     self.y = self.y_train[indices[0:1000], :]
        # else:
        #     self.X = self.X_val
        #     self.y = self.y_val
        self.tsne = sklearn.manifold.TSNE(n_components=self.n_components_tsne,
                                          perplexity=self.perplexity,
                                          learning_rate=self.learning_rate,
                                          metric=self.metric,
                                          verbose=True)
        # self.pca = sklearn.decomposition.PCA(n_components=self.n_components_pca)


    def reduce(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, "rb") as fh:
                self.X_emb, self.y = pickle.load(fh)
        else:
            # X_pca = self.pca.fit_transform(self.X)
            X_tsne = self.tsne.fit_transform(self.X)
            self.X_emb = X_tsne
            with open(self.save_path, "wb") as fh:
                pickle.dump((self.X_emb, self.y), fh)

    def visualize(self):
        plt.figure()
        plt.grid(True)
        for cl in self.classes:
            ind = self.y.flatten() == cl
            # pdb.set_trace()
            x = self.X_emb[ind, :]
            plt.scatter(x[:, 0], x[:, 1], alpha=0.2, edgecolors='none', label=self.classes[cl])
        plt.legend()
        plt.show()

    def _binarize(self, y, cl):
        _bin = copy(y)
        if cl == 0:
            _bin[_bin == 0] = -1
            _bin[_bin > 0] = 0
            _bin[_bin == -1] = 1
        else:
            _bin *= (_bin == cl)
            _bin[_bin != 0] = 1
        return _bin

    def _load_data(self):
        if not (os.path.exists(os.path.join(self.train_dset.root, 'X_train.npy') or os.path.exists(os.path.join(self.train_dset.root, 'y_train.npy')))):
            print("Converting Train data to numpy")
            self.X_train, self.y_train = self._gen_numpy_data(self.train_dset)
            np.save(os.path.join(self.train_dset.root, 'X_train.npy'), self.X_train)
            np.save(os.path.join(self.train_dset.root, 'y_train.npy'), self.y_train)
        else:
            self.X_train = np.load(os.path.join(self.train_dset.root, 'X_train.npy'))
            self.y_train = np.load(os.path.join(self.train_dset.root, 'y_train.npy'))

        if not (os.path.exists(os.path.join(self.val_dset.root, 'X_val.npy') or os.path.exists(os.path.join(self.val_dset.root, 'y_val.npy')))):
            print("Converting Val data to numpy")
            self.X_val, self.y_val = self._gen_numpy_data(self.val_dset)
            np.save(os.path.join(self.val_dset.root, 'X_val.npy'), self.X_val)
            np.save(os.path.join(self.val_dset.root, 'y_val.npy'), self.y_val)
        else:
            self.X_val = np.load(os.path.join(self.val_dset.root, 'X_val.npy'))
            self.y_val = np.load(os.path.join(self.val_dset.root, 'y_val.npy'))

        if not (os.path.exists(os.path.join(self.test_dset.root, 'X_test.npy') or os.path.exists(os.path.join(self.test_dset.root, 'y_test.npy')))):
            print("Converting Test data to numpy")
            self.X_test, self.y_test = self._gen_numpy_data(self.test_dset)
            np.save(os.path.join(self.test_dset.root, 'X_test.npy'), self.X_test)
            np.save(os.path.join(self.test_dset.root, 'y_test.npy'), self.y_test)
        else:
            self.X_test = np.load(os.path.join(self.test_dset.root, 'X_test.npy'))
            self.y_test = np.load(os.path.join(self.test_dset.root, 'y_test.npy'))

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
