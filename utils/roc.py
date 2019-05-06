from utils.metric import Metric
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from copy import copy
from math import ceil

class ROC(Metric):
    def __init__(self, **kwargs):
        self.metric_name = 'ROC'
        self.filename = kwargs.get('filename', 'plot.png')
        self.classes = kwargs.get('classes', [0,1,2,3,4,5,6])
        self.class_names = kwargs.get('class_names', ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
        self.y_true = kwargs.get('y_true', [])
        self.y_pred = kwargs.get('y_pred', [])
        self.model_name = kwargs.get('model_name', self.filename)

    def compute_values(self, y_true, y_pred):
        roc_auc = roc_auc_score(y_true, y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
        return fpr, tpr, thresholds, roc_auc

    def metric_plot(self):
        for cl in self.classes:
            ax = plt.subplot(ceil(len(self.classes)/2), 2, cl+1)
            _bin_labels = self._binarize(self.y_true, cl)
            _bin_preds = self._binarize(self.y_pred, cl)
            fpr, tpr, thresholds, roc_auc = self.compute_values(_bin_labels, _bin_preds)
            ax.plot(fpr, tpr)
            plt.title("%s: ROC for %s vs all | AUC = %.3f" %(self.model_name, self.class_names[cl], roc_auc))

        plt.tight_layout()
        Metric.saveFigure(self, plt, self.filename)
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
