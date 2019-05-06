from utils.nn_embeddings import NNEmbeddings
from utils.face_detector import FRDetector
import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np

if __name__ == '__main__':
    fd = FRDetector()
    encoder = NNEmbeddings()
    imgname1 = 'images/cp1.jpg'
    img1 = imread(imgname1)
    faces, locations = fd.get_faces(img1)
    embeddings = encoder.get_embeddings(img1, face_locations=locations)
    print(embeddings)
