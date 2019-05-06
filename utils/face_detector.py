import face_recognition
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from abc import ABC, abstractmethod

class FaceDetector(ABC):
    def __init__(self, detector=None, resize=(48, 48, 1)):
        self.detector = detector
        self.resize = resize

    @abstractmethod
    def get_faces(self):
        raise NotImplementedError

class FRDetector(FaceDetector):
    def __init__(self, detector="FR", resize=(48, 48)):
        super(FRDetector, self).__init__(detector=detector, resize=resize)

    def _detect(self, img):
        face_locations = face_recognition.face_locations(img)
        return face_locations

    def get_faces(self, img=None):
        locs = self._detect(img)
        faces = []
        for loc in locs:
            face = self._crop(img, loc)
            if self.resize is not None:
                face = imresize(face, self.resize)
            faces.append(face)
        return faces, locs

    def _crop(self, img, loc):
        top, right, bottom, left = loc
        if img.ndim > 2:
            return img[top:bottom, left:right, :]
        else:
            return img[top:bottom, left:right]

if __name__ == '__main__':
    fd = FRDetector()
    imgname = 'img.jpeg'
    img = imread(imgname)
    faces = fd.get_faces(img)
    plt.figure()
    for i, face in enumerate(faces):
        plt.subplot(1, len(faces), i+1)
        plt.imshow(face)
    plt.show()
