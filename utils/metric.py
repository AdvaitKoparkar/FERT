from abc import ABC, abstractmethod
import os
import matplotlib.pyplot as plt

class Metric(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def compute_values(self):
        raise NotImplementedError

    @abstractmethod
    def metric_plot(self):
        raise NotImplementedError

    def saveFigure(self, fig, filename):
        file_path=os.path.join('result_plots',filename)
        if not os.path.exists('result_plots'):
            os.mkdir('result_plots')
        fig.savefig(file_path)
        print("Saved Figure as %s " %file_path)
