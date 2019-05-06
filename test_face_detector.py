from utils.face_detector import FRDetector
from scipy.misc import imread
import matplotlib.pyplot as plt
import pdb

if __name__ == '__main__':
    fd = FRDetector()
    imgname = 'images/dg.jpg'
    img = imread(imgname)
    # pdb.set_trace()
    faces, _ = fd.get_faces(img)
    plt.figure()
    for i, face in enumerate(faces):
        plt.subplot(1, len(faces), i+1)
        plt.imshow(face)
    plt.show()
