from experiments.classification import Classification
from dataloaders.fer_loader import FERDataset

from models.alex_net import AlexNet
import pickle
from scipy.misc import imread
from utils.face_detector import FRDetector
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Training and saving
    classification_cnn = Classification(**{'classifier': AlexNet, 'dset':FERDataset, 'max_samples': 1000, 'neural_net':True})
    classification_cnn.grid_search()
    classification_cnn.gen_metrics()
