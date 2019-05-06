import itertools
from tqdm import tqdm
import os
import numpy as np
import pickle
from torch.utils.data import DataLoader

class GridSearch(object):
    def __init__(self, model, train_dset, val_dset, test_dset, hyper_params, neural_net=False, train_writer=None, resume=False, **kwargs):
        self.model = model
        self.resume = resume
        self.train_dset = train_dset
        self.val_dset = val_dset
        self.test_dset = test_dset
        self.neural_net = neural_net
        self.train_writer = train_writer
        self.save_path = kwargs.get('save_path', None)
        if self.save_path is None:
            self.save_path = os.path.join('runs', model.__name__+'.pkl')
        if not os.path.exists('runs'):
            os.mkdir('runs')
        # Get data in numpy format
        self._load_data()

        if not self.neural_net:
            keys, values = zip(*hyper_params.items())
            self.configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
        self.best_score = -1
        self.best_config = None
        self.best_model = None
        self.step = 0

    def search(self):
        if self.neural_net is False:
            for config in tqdm(self.configs):
                _model = self.model(self.train_dset, **config)
                _model.train()
                _model_score = _model.score(self.X_val, self.y_val)
                if _model_score >= self.best_score:
                    self.best_score = _model_score
                    self.best_config = config
                    self.best_model = _model
        else:
            if self.resume is True and os.path.exists(self.save_path):
                print("Resuming training from %s" %self.save_path)
                with open(self.save_path, 'rb') as fh:
                    model_dict = pickle.load(fh)
                _model = self.model(self.train_dset, val_dset=self.val_dset, test_dset=self.test_dset, train_writer=self.train_writer, step=model_dict['step'])
                _model.set_state(model_dict['model'])
            else:
                _model = self.model(self.train_dset, val_dset=self.val_dset, test_dset=self.test_dset, train_writer=self.train_writer, step=0)
            _model.train()
            val_dataloader = DataLoader(self.val_dset, batch_size=_model.batch_size, shuffle=False, num_workers=1)
            self.best_score = _model.score(dloader=val_dataloader)
            self.config = None
            self.best_model = _model.get_state()
            self.step = _model.step

        # save_path = os.path.join('runs', _model.model_name+'.pkl')
        print("Saving best model in: %s" %(self.save_path))
        save_state = self._best_model()
        with open(self.save_path, 'wb') as fh:
            pickle.dump(save_state, fh)
        return save_state

    def _best_model(self):
        if self.neural_net:
            return {'score': self.best_score, 'model': self.best_model, 'step':self.step}
        else:
            return {'score': self.best_score, 'model': self.best_model, 'config':self.best_config}


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
