from models.dbscan import DBSCAN
from models.mean_shift import MeanShift
from utils.face_detector import FRDetector
from scipy.misc import imread
import matplotlib.pyplot as plt
import pdb

if __name__ == '__main__':
    fd = FRDetector()
    cluster = MeanShift()
    img1 = imread('images/cp1.jpg')
    img2 = imread('images/cp2.jpg')
    img3 = imread('images/img.jpeg')
    img4 = imread('images/cp3.jpg')
    img5 = imread('images/cp4.jpg')
    img6 = imread('images/cp5.jpeg')
    img7 = imread('images/cp6.jpeg')
    faces = []
    faces1, loc1 = fd.get_faces(img1)
    faces2, loc2 = fd.get_faces(img2)
    faces3, loc3 = fd.get_faces(img3)
    faces4, loc4 = fd.get_faces(img4)
    faces5, loc5 = fd.get_faces(img5)
    faces6, loc6 = fd.get_faces(img6)
    faces7, loc7 = fd.get_faces(img7)
    imgs = [img1,img2,img3,img4,img5,img6,img7]
    faces = faces1+faces2+faces3+faces4+faces5+faces6+faces7
    locs = [loc1,loc2,loc3,loc4,loc5,loc6,loc7]
    # labels = cluster.cluster(frames=[img1, img2, img3, img4, img5, img6, img7], locations=locs)
    # labels = cluster.cluster(frames=[img1, img2, img3, img4, img5, img6, img7])
    loc_force = [[(0, 47, 47, 0)]]*len(faces)
    labels, hashes = cluster.cluster(faces)
    print(labels)
    plt.figure()
    for i, face in enumerate(faces):
        plt.subplot(1, len(faces), i+1)
        plt.imshow(face)
        plt.title('Person %d' %labels[i])
    plt.show()
