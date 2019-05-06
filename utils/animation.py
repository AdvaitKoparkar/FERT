import cv2
import os
import numpy as np
import pickle

class Animation(object):
    def __init__(self, **kwargs):
        self.ip = kwargs.get('video_path', os.path.join("runs", "op.avi"))
        self.cluster_path = kwargs.get('results_path', os.path.join("runs", "clustered.pkl"))
        self.op = kwargs.get('final_path', os.path.join("runs", "fert.avi"))
        self.cap = cv2.VideoCapture(self.ip)
        frame_width,frame_height = 640, 480
        self.video_writer = cv2.VideoWriter(self.op, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
        with open(self.cluster_path, "rb") as fh:
            self.fer_cluster = pickle.load(fh)

    def make(self):
        frame_idx = -1
        while True:
            frame_idx += 1
            _, frame = self.cap.read()
            if frame is None:
                break
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_rects = {}
            for person in self.fer_cluster.keys():
                if frame_idx in self.fer_cluster[person]['frame_count']:
                    frame_no = self.fer_cluster[person]['frame_count'].index(frame_idx)
                    bb = self.fer_cluster[person]['locations'][frame_no]
                    exp = self.fer_cluster[person]['expressions'][frame_no]
                    frame_rects[person] = (person, bb, exp)
            for k in frame_rects.keys():
                person, bb, exp = frame_rects[k]
                text = "Person %d: %s" %(person, exp)
                pt1, pt2 = (bb[3], bb[0]), (bb[1], bb[2])
                cv2.rectangle(frame, pt1=pt1, pt2=pt2, color=(255,255,255))
                cv2.putText(frame, text, pt1, fontFace=0, fontScale=1, color=(255,255,255))

            # pdb.set_trace()
            self.video_writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.video_writer.release()
