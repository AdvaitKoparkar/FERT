from utils.metric import Metric
import matplotlib.pyplot as plt
import pdb
import numpy as np
from sklearn.metrics import accuracy_score

class Accuracy(Metric):
    def __init__(self, **kwargs):
        self.metric_name='Accuracy'
        self.filename=kwargs.get('filename','acc.png')
        self.classes = kwargs.get('classes',[0,1,2,3,4,5,6])
        self.class_names=kwargs.get('class_names', ['Angry','Disgust','Fear','Happy', 'Sad', 'Surprise','Neutral'])
        self.y_true=kwargs.get('y_true',[])
        self.y_pred =kwargs.get('y_pred',[])

    def compute_values(self):
        total_acc=accuracy_score(self.y_true, self.y_pred)
        print("Accuracy is :", total_acc)
        class_acc=np.zeros(len(self.classes))
        for cl in self.classes:
            cl_pred=self.y_pred==cl*np.ones((len(self.y_pred)))
            cl_true=(self.y_true)==cl*np.ones((len(self.y_true)))
            class_acc[cl]=accuracy_score(cl_pred, cl_true)
        return(total_acc, class_acc)

    def metric_plot(self):
        [acc, class_acc]=self.compute_values()
        plt.figure()
        plt.bar(self.classes, class_acc)
        plt.title("%s Total Accuracy" %acc)
        plt.show()
        Metric.saveFigure(self,plt,self.filename)
