import cv2
import numpy as np
from Config import Config

import sys
sys.path.append(r'./Thirdparty/pyfbow/build/pyfbow.so')
import pyfbow

from scipy.spatial.transform import Rotation


class Frame:
    id = 0

    def __init__(self, imRGB, imD, timestamp, bgr=True, add_noise=False):
        if bgr:
            im_rgb = cv2.cvtColor(imRGB, cv2.COLOR_BGRA2GRAY)
        else:
            im_rgb = cv2.cvtColor(imRGB, cv2.COLOR_RGBA2GRAY)
        self.timestamp = timestamp
        self.id = Frame.id
        self.fp = []
        self.depth = []
        Frame.id += 1
        self.height, self.width = im_rgb.shape[:2]

        self.pose = None  # np.array([0, 0, 0, 0, 0, 0, 1])

        self.feature_per_block = []
        # Initiate ORB detector
        orb = cv2.ORB_create(nfeatures=Config().n_features,
                             scaleFactor=Config().scale_factor,
                             nlevels=Config().n_levels,
                             fastThreshold=Config().th_fast)  # 1500, 31
        # find the keypoints and descriptors with ORB
        for x1, x2, y1, y2 in self.blocks(1, 1):
            fps = orb.detect(im_rgb[y1:y2, x1:x2], None)
            # for feature_point in fps:
            for idx in range(len(fps) - 1, -1, -1):
                feature_point = fps[idx]
                yy = np.int(np.rint(feature_point.pt[1]))
                xx = np.int(np.rint(feature_point.pt[0]))
                depth = imD[yy][xx]
                if depth > 0.05:
                    feature_point.pt = (feature_point.pt[0] + x1, feature_point.pt[1] + y1)
                    if add_noise:
                        depth += np.random.normal(0, 0.2*depth, 1)[0]
                    self.depth.append(depth)
                else:
                    fps.pop(idx)

            self.fp.extend(fps)
            self.feature_per_block.append(len(fps))

        self.fp, self.des = orb.compute(im_rgb, self.fp)
        self.depth = np.zeros_like(self.fp, dtype=np.float32)
        for idx in range(len(self.fp)):
            yy = np.int(np.rint(self.fp[idx].pt[1]))
            xx = np.int(np.rint(self.fp[idx].pt[0]))
            self.depth[idx] = imD[yy][xx]

        print("Frame ", self.id, " #features: ", len(self.fp), " : ", self.feature_per_block)
        # self.draw_points(imRGB)

    def get_id(self):
        return self.id

    def draw_points(self, imRGB):
        if not self.fp:
            print("No key points detected")
        else:
            img_test = cv2.drawKeypoints(imRGB, self.fp, None, color=(0, 255, 0), flags=0)
            # cv2.imshow('frame ' + str(self.id), img_test)
            return img_test

    def blocks(self, rows, cols):
        xs = np.uint32(np.rint(np.linspace(0, self.width, num=cols+1)))
        ys = np.uint32(np.rint(np.linspace(0, self.height, num=rows+1)))
        y_starts, y_ends = ys[:-1], ys[1:]
        x_starts, x_ends = xs[:-1], xs[1:]
        for y1, y2 in zip(y_starts, y_ends):
            for x1, x2 in zip(x_starts, x_ends):
                yield x1, x2, y1, y2  # self.imRGB[y1:y2, x1:x2]

    def rot(self):
        if self.pose:
            return Rotation.from_quat(self.pose[3:])
        return None

    def pos(self):
        if self.pose:
            return np.array(self.pose[0:3])
        return None

    @classmethod
    def get_point_cloud(cls, imRGB, imD, flag_rgb=True, depth_th=None):
        import open3d as o3d
        fx = Config().fx
        fy = Config().fy
        cx = Config().cx
        cy = Config().cy

        width = imRGB.shape[1]
        height = imRGB.shape[0]

        n_points = width * height

        points = np.zeros((n_points, 3))
        colors = np.zeros((n_points, 3))
        counter = 0
        for idx in range(width):
            for jdx in range(height):
                d = imD[jdx, idx]
                if d < 0.1:
                    continue
                if depth_th:
                    if d > depth_th:
                        continue
                points[counter, 0] = (idx-cx)*d/fx
                points[counter, 1] = (jdx-cy)*d/fy
                points[counter, 2] = d
                if flag_rgb is not None:
                    colors[counter, :] = imRGB[jdx, idx, 0:3] / 256.0
                else:
                    colors[counter, :] = imRGB[jdx, idx, 3::-1] / 256.0
                counter += 1

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([pcd])
        return pcd, points

    @classmethod
    def save(cls, rgb_, depth_, save_dir, name):
        from PIL import Image
        import os
        # name = str(self.id)
        im = Image.fromarray(rgb_, mode="RGBA")
        im.save(os.path.join(save_dir, name+'.png'))
        np.save(os.path.join(save_dir, name+'.npy'), depth_)
