from utils.metric import Metric
import matplotlib.pyplot as plt
import pdb
import numpy as np
from sklearn.metrics import confusion_matrix

class confusionMatrix(Metric):
    def __init__(self, **kwargs):
        self.metric_name='ConfusionMatrix'
        self.filename=kwargs.get('filename','confmat.png')
        self.classes = kwargs.get('classes',[0,1,2,3,4,5,6])
        self.class_names=kwargs.get('class_names', ['Angry','Disgust','Fear','Happy', 'Sad', 'Surprise','Neutral'])
        self.y_true=kwargs.get('y_true',[])
        self.y_pred =kwargs.get('y_pred',[])

    def compute_values(self):
        conf_mat=confusion_matrix(self.y_true, self.y_pred)
#        print("Confusion Matrix is :", conf_mat)
        return(conf_mat)

    def metric_plot(self):
        cm=self.compute_values()
        
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=self.class_names, yticklabels=self.class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
        
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
#        pdb.set_trace()
        Metric.saveFigure(self,plt,self.filename)

