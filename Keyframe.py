from Frame import Frame
import numpy as np
import pyfbow
from KeyframeDatabase import KeyframeDatabase
from Config import Config
from Keypoint import KeyPoint

from scipy.spatial.transform import Rotation


class Keyframe:
    kfID = 0
    voc = pyfbow.Vocabulary()
    voc.readFromFile(Config().bow)
    kfdb = KeyframeDatabase()

    def __init__(self, frame):
        self.timestamp = frame.timestamp

        self.kfID = Keyframe.kfID
        self.fID = frame.id
        Keyframe.kfID += 1

        self.fp, self.des = frame.fp, frame.des
        self.depth = frame.depth

        self.height, self.width = frame.height, frame.width

        self.pose = frame.pose
        # self.rot = frame.rot
        # self.pos = frame.pos

        self.neighbors = []

        # self.bow = []
        # self.bow_ind = []
        # def compute_bow(self):
        # Extract bow vector from features, and the indices of the words at the 4th lvl of tree
        self.bow, bow_ind = Keyframe.voc.transform2(self.des, 4)
        self.bow_ind = bow_ind.keys()

        Keyframe.kfdb.insert(self)

        self.key_points = dict()
        self.n_kp = 0  # Number of keypoints

        self.covariance = None

    def key_point_initializer(self):
        fx = Config().fx
        fy = Config().fy
        cx = Config().cx
        cy = Config().cy
        for idx in range(len(self.fp)):
            d = self.depth[idx]
            pos = [(self.fp[idx].pt[0] - cx) / fx * d, (self.fp[idx].pt[1] - cy) / fy * d, d]
            self.key_points[idx] = KeyPoint(pos, self.des[idx])
        self.n_kp = len(self.fp)

    def add_key_point(self, idx, kp):  # idx of the feature point, and kp
        if idx not in self.key_points:  # if kp is not in this
            self.key_points[idx] = kp
            self.n_kp += 1
            return True
        else:
            assert kp.id == self.key_points[idx].id  # .get_id()
            return False

    def new_key_point(self, idx, pos):
        if idx not in self.key_points:
            new_kp = KeyPoint(pos, self.des[idx])
            self.key_points[idx] = new_kp
            self.n_kp += 1
            return new_kp
        else:
            return self.key_points[idx]

    def get_key_point(self, idx):
        if idx in self.key_points:
            return self.key_points[idx]
        else:
            return None

    def set_pose(self, rot, trans):
        pose = np.eye(4)
        pose[0:3, 0:3] = rot
        pose[0:3, 3] = trans
        self.pose = pose

    def rot(self):
        # if self.pose:
        #     return Rotation.from_quat(self.pose[3:]).as_matrix()
        return self.pose[0:3, 0:3]
        # return None

    def pos(self):
        # if self.pose:
        #     return np.array(self.pose[0:3])
        return self.pose[0:3, 3]
        # return None

    def pose_matrix(self):
        # r = self.rot()
        # t = self.pos()
        # tt = np.expand_dims(t, axis=1)
        # return np.concatenate((np.concatenate((r, tt), axis=1),
        #                        np.array([0, 0, 0, 1], ndmin=2)), axis=0)
        return self.pose

    def neighbors_list(self):
        return [kf for kf, _, _ in self.neighbors]

