from experiments.classification import Classification
from dataloaders.fer_loader import FERDataset
from models.simple_cnn import SimpleCNN
from models.shufflenet import ShuffleNet
from models.tinyresnet import TinyResNet
from models.mini_xception import MiniXception
import pickle
from scipy.misc import imread
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Training and saving
    classification_cnn = Classification(**{'classifier': SimpleCNN, 'dset':FERDataset, 'neural_net':True, 'save_path': 'runs/SimpleCNN_tsne_test.pkl'})
    classification_cnn.grid_search()
    classification_cnn.gen_metrics()
