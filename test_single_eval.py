from dataloaders.fer_loader import FERDataset
from utils.face_detector import FRDetector
from experiments.classification import Classification
from models.simple_cnn import SimpleCNN
from models.knn import KNNClassifier
from models.svm import SVMClassifier
from scipy.misc import imread

if __name__ == "__main__":
    classification_cnn = Classification(**{'classifier': SimpleCNN, 'dset':FERDataset, 'neural_net':True})
    classification_knn = Classification(**{'classifier': KNNClassifier, 'dset':FERDataset, 'max_samples': 1000})
    classification_svm = Classification(**{'classifier':SVMClassifier, 'dset': FERDataset})
    fd = FRDetector()
    img1 = imread('images/cp1.jpg')
    faces1, loc1 = fd.get_faces(img1)
    face = faces1[0][:, :, 0:1]
    print(face.shape)
    print(classification_cnn.single_eval(face))
    print(classification_knn.single_eval(face))
    print(classification_svm.single_eval(face))
