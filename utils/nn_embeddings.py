from abc import ABC, abstractmethod
import numpy as np
import face_recognition
from utils.embeddings import Embeddings
import cv2

class NNEmbeddings(Embeddings):
    def __init__(self, **kwargs):
        self.name = "NNEmbeddings"
        self.num_jitters = kwargs.get('num_jitters', 1)

    def get_embeddings(self, faces, face_locations=None):
        if isinstance(faces, list):
            embeddings_list = []
            for face in faces:
                if face.ndim < 3:
                    face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
                embeddings_list += face_recognition.face_encodings(face, num_jitters=self.num_jitters)
            return embeddings_list
        else:
            if faces.ndim < 3:
                faces = cv2.cvtColor(faces, cv2.COLOR_GRAY2BGR)
            return face_recognition.face_encodings(faces, known_face_locations=face_locations)
