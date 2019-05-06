import cv2
from dataloaders.fer_loader import FERDataset
from utils.face_detector import FRDetector
from experiments.classification import Classification
from models.simple_cnn import SimpleCNN
from models.shufflenet import ShuffleNet
from models.tinyresnet import TinyResNet
from experiments.ensemble import Ensemble
from models.dbscan import DBSCAN
from utils.video import Video

if __name__ == "__main__":
	classifiers = [{'classifier': ShuffleNet, 'dset':FERDataset, 'max_samples': 1000, 'neural_net':True},
	{'classifier': SimpleCNN, 'dset':FERDataset, 'max_samples': 1000, 'neural_net':True},
	{'classifier': TinyResNet, 'dset':FERDataset, 'max_samples': 1000, 'neural_net':True}]

	hyper_params = [None,
	                None,
	                None]

	neural_net = [True,
	              True,
	              True]

	ensemble = Ensemble(classifiers=classifiers,
	                    hyper_params=hyper_params,
	                    classifier_type=neural_net)

	video = Video(**{'detector': FRDetector(),
	                 'classification': ensemble,
	                 'cluster': DBSCAN()
	                  })
	video.run()
