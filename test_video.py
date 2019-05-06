import cv2
from dataloaders.fer_loader import FERDataset
from utils.face_detector import FRDetector
from experiments.classification import Classification
from models.simple_cnn import SimpleCNN
from models.dbscan import DBSCAN
from utils.video import Video

if __name__ == "__main__":
    video = Video(**{'detector': FRDetector(),
                     'classification': Classification(**{'classifier': SimpleCNN, 'dset':FERDataset, 'neural_net':True}),
                     'cluster': DBSCAN(),
                     'capture': True
                      })
    video.run()
