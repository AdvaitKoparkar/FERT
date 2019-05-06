import cv2
from dataloaders.fer_loader import FERDataset
from utils.face_detector import FRDetector
from experiments.classification import Classification
from models.simple_cnn import SimpleCNN
from models.dbscan import DBSCAN
import pickle
import os
from scipy.misc import imread

class Video(object):
    def __init__(self, **kwargs):
        self.capture = kwargs.get('capture', True)
        if self.capture:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture("data/vid.mp4")
        self.detector = kwargs.get('detector', FRDetector())
        self.classification = kwargs.get('classification', Classification(**{'classifier': SimpleCNN, 'dset':FERDataset, 'neural_net':True}))
        self.cluster = kwargs.get('cluster', DBSCAN())
        self.max_frames = kwargs.get('max_frames', 50)
        self.results_path = kwargs.get('results_path', os.path.join("runs", "clustered.pkl"))
        self.video_path = kwargs.get('frames_path', os.path.join("runs", "video.pkl"))
        self.op_video = kwargs.get('video_path', os.path.join("runs", "op.avi"))
        self.frame_dict = {}
        self.all_faces = []
        self.frames = []
        self.fer_cluster = {}
        frame_width,frame_height = 640, 480
        self.video_writer = cv2.VideoWriter(self.op_video, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    def run(self):
        self._fer()
        self._cluster()


    def _fer(self):
        frame_count = 0
        while True:
            if frame_count >= self.max_frames:
                break
            ret, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces, loc = self.detector.get_faces(frame)
            self.all_faces += faces
            expressions = []
            hashes = []
            if len(faces) > 0:
                self.frames.append(frame)
                self.video_writer.write(frame)
                self.frame_dict[frame_count] = {}
                for face_idx, face in enumerate(faces):
                    hashes.append(hash(face.tostring()))
                    expressions.append(self.classification.train_dset.classes[self.classification.single_eval(face)[0][0]])
                self.frame_dict[frame_count]['faces'] = faces
                self.frame_dict[frame_count]['locations'] = loc
                self.frame_dict[frame_count]['expressions'] = expressions
                self.frame_dict[frame_count]['hashes'] = hashes
                frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        self.video_writer.release()


    def _cluster(self):
        labels, hashes = self.cluster.cluster(self.all_faces)
        for idx in range(len(labels)):
            h = hashes[idx]
            l = labels[idx]
            self.fer_cluster[l] = {}
            self.fer_cluster[l]['faces'] = []
            self.fer_cluster[l]['locations'] = []
            self.fer_cluster[l]['expressions'] = []
            self.fer_cluster[l]['frame_count'] = []
        for idx in range(len(labels)):
            h = hashes[idx]
            l = labels[idx]
            for frame_idx in self.frame_dict.keys():
                for fi, hc in enumerate(self.frame_dict[frame_idx]['hashes']):
                    if hc == h:
                        self.fer_cluster[l]['faces'].append(self.frame_dict[frame_idx]['faces'][fi])
                        self.fer_cluster[l]['expressions'].append(self.frame_dict[frame_idx]['expressions'][fi])
                        self.fer_cluster[l]['locations'].append(self.frame_dict[frame_idx]['locations'][fi])
                        self.fer_cluster[l]['frame_count'].append(frame_idx)

        print("Saving all frames to %s" %self.video_path)
        with open(self.video_path, "wb") as fh:
            pickle.dump(self.frames, fh)
        print("Saving clustered faces to %s" %self.results_path)
        with open(self.results_path, "wb") as fh:
            pickle.dump(self.fer_cluster, fh)

        print(self.fer_cluster[0]['expressions'])
        return self.fer_cluster
