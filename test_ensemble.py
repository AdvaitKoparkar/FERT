from experiments.classification import Classification
from dataloaders.fer_loader import FERDataset
from experiments.ensemble import Ensemble
from models.logreg import LogRegClassifier
from models.simple_cnn import SimpleCNN
from models.shufflenet import ShuffleNet
from models.tinyresnet import TinyResNet
from scipy.misc import imread
from utils.face_detector import FRDetector

if __name__ == '__main__':
    classifiers = [{'classifier': ShuffleNet, 'dset':FERDataset, 'max_samples': 1000, 'neural_net':True},
    {'classifier': TinyResNet, 'dset':FERDataset, 'max_samples': 1000, 'neural_net':True},
    {'classifier': TinyResNet, 'dset':FERDataset, 'max_samples': 1000, 'neural_net':True}]

    hyper_params = [None,
                    None,
                    None]

    neural_net = [True,
                  True,
                  True]

    ensemble = Ensemble(classifiers=classifiers,
                        hyper_params=hyper_params,
                        classifier_type=neural_net,
                        model_weights=[None, "runs/tiny_resnet_100_epochs.pkl", None])
    ensemble.grid_search(restore=True)
    ensemble.gen_metrics()
    print(ensemble.score())
    # img = imread('images/cp1.jpg')
    # fd = FRDetector()
    # faces1, loc1 = fd.get_faces(img)
    # print(ensemble.score(faces1[0]))
