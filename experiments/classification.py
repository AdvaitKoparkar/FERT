from utils.grid_search import GridSearch
import numpy as np
from tqdm import tqdm
from utils.f1_score import F1_score
from utils.roc import ROC
from utils.accuracy import Accuracy
from utils.confusionMatrix import confusionMatrix
import os
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import pickle
import pdb


class Classification(object):
    def __init__(self, **kwargs):
        self.classifier = kwargs.get('classifier', None)
        self.dset_name = kwargs.get('dset', None)
        self.dset_root = kwargs.get('dset_root', './data/fer2013')
        self.dset_filename = kwargs.get('dset_filename', 'fer2013.csv')
        self.searcher = kwargs.get('searcher', GridSearch)
        self.save_path = kwargs.get('save_path', None)
        # Is the classifier a neutal network?
        self.neural_net = kwargs.get('neural_net', False)
        self.best_model = {}

        # Set up train and test datasets
        if not callable(self.dset_name):
            raise Exception("Invalid dataset")
        self.train_dset = self.dset_name(root=self.dset_root, filename=self.dset_filename, mode='train', tensor=self.neural_net)
        self.val_dset = self.dset_name(root=self.dset_root, filename=self.dset_filename, mode='val', tensor=self.neural_net)
        self.test_dset = self.dset_name(root=self.dset_root, filename=self.dset_filename, mode='test', tensor=self.neural_net)
        self._load_data()
        if self.neural_net:
            self.train_writer = SummaryWriter('runs/'+str(self.classifier.__name__))
        else:
            self.train_writer = None

    def embeddings(self):
        train_dataloader = DataLoader(self.train_dset, batch_size=4, shuffle=False, num_workers=1)
        if self.save_path is None:
            save_path = os.path.join('runs', self.classifier.__name__+'.pkl')
        else:
            save_path = self.save_path
        with open(save_path, "rb") as fh:
            saved_model = pickle.load(fh)
        if self.neural_net is True:
            # pdb.set_trace()
            _model = self.classifier(self.train_dset)
            _model.set_state(saved_model['model'])
            return _model.embeddings(train_dataloader)

    def get_data(self):
        return self.y_test

    def restore_classifier(self, fpath=None):
        if fpath is None:
            with open(os.path.join('runs', self.classifier.__name__+'.pkl'), 'rb') as fh:
                self.best_model = pickle.load(fh)
        else:
            with open(fpath, 'rb') as fh:
                self.best_model = pickle.load(fh)

    def grid_search(self, hyper_params=None):
        self.hyper_params = hyper_params
        searcher_obj = self.searcher(model=self.classifier, hyper_params=hyper_params, train_dset=self.train_dset, val_dset=self.val_dset, test_dset=self.test_dset, neural_net=self.neural_net, train_writer=self.train_writer, save_path=self.save_path)
        self.best_model  = searcher_obj.search()

    # Use grid search instead
    # def train_cnn(self):
    #     model = self.classifier(self.train_dset, val_dset=self.val_dset, test_dset=self.test_dset, train_writer=self.train_writer)
    #     model.train()

    def visualize_best_model(self):
        preds = self.best_model['model'].test(self.X_test)
        for idx, x in enumerate(self.X_test):
            img = x.reshape(self.test_dset.size)
            pred_exp = self.test_dset.classes[preds[idx]]
            true_exp = self.test_dset.classes[self.y_test[idx,0]]
            plt.imshow(img)
            plt.title('Predicted %s | True %s' %(pred_exp, true_exp))
            plt.show()

    def eval(self):
        if self.neural_net is False:
            y_pred = self.best_model['model'].eval(X=self.X_test)[:, None]
        else:
            _model = self.classifier(self.train_dset)
            _model.set_state(self.best_model['model'])
            test_dataloader = DataLoader(self.test_dset, batch_size=_model.batch_size, shuffle=False, num_workers=1)
            y_pred = _model.eval(dloader=test_dataloader)
        return y_pred

    def gen_metrics(self):
        if self.neural_net is False:
            y_pred = self.best_model['model'].eval(X=self.X_test)[:, None]
            fname = self.best_model['model'].model_name
        else:
            _model = self.classifier(self.train_dset)
            _model.set_state(self.best_model['model'])
            test_dataloader = DataLoader(self.test_dset, batch_size=_model.batch_size, shuffle=False, num_workers=1)
            y_pred = _model.eval(dloader=test_dataloader)
            # y_pred = _model.eval(X=self.X_test)
            fname = _model.model_name
        labels = self.y_test
        f1_score = F1_score(**{'filename':fname+'_f1.png', 'y_true':labels, 'y_pred': y_pred, 'model_name': fname})
        roc = ROC(**{'filename':fname+'_roc.png', 'y_true':labels, 'y_pred': y_pred, 'model_name': fname})
        acc=Accuracy(**{'filename':fname+'_acc.png', 'y_true':labels, 'y_pred': y_pred, 'model_name': fname})
        conf_mat=confusionMatrix(**{'filename':fname+'_confmat.png', 'y_true':labels, 'y_pred': y_pred, 'model_name': fname})

        self._plot_metrics([f1_score, roc, acc, conf_mat])
        

    def _plot_metrics(self, metric_list):
        for metric in metric_list:
            metric.metric_plot()

    def single_eval(self, X):
        if self.save_path is None:
            save_path = os.path.join('runs', self.classifier.__name__+'.pkl')
        else:
            save_path = self.save_path
        with open(save_path, "rb") as fh:
            saved_model = pickle.load(fh)
        if self.neural_net is True:
            # pdb.set_trace()
            _model = self.classifier(self.train_dset)
            _model.set_state(saved_model['model'])
            return _model.eval(X=X)[0]
        else:
            return saved_model['model'].eval(X=X)

    def _load_data(self):
        if not (os.path.exists(os.path.join(self.test_dset.root, 'X_test.npy') or os.path.exists(os.path.join(self.test_dset.root, 'y_test.npy')))):
            print("Converting test data to numpy")
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
