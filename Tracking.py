import numpy as np
import cv2
import open3d as o3d
from Config import Config
from matplotlib import pyplot as plt
from Optimizer import *
from Keyframe import *
from utilities import rot_to_angle, rot_to_heading

from scipy.spatial.transform import Rotation


class Tracking:
    """Track the input image with respect to previous images"""
    """fx=fy=f=imageWidth /(2 * tan(CameraFOV * π / 360))"""
    def __init__(self):
        self.current_frame = None
        self.ref_keyframe = None
        self.image_queue = []  # ?
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # create BFMatcher object

        self.insert_new_kf = False
        self.new_kf_inserted = False
        self.tracking_success = False

        self.map = []
        # self.current_rot = np.eye(3)
        # self.current_pos = np.array([0, 0, 0])
        self.current_pose = np.eye(4)

        self.fx = Config().fx
        self.fy = Config().fy
        self.cx = Config().cx
        self.cy = Config().cy
        self.bf = Config().bf

        self.reprojection_threshold = 0.3

        self.n_loop_closures = 0    # Counter for the number of verified loop closures

        self.grid_dim = (31, 31, 10)  # x_grid, z_grid, [x,z,heading uncertainty + 1 layer occupancy]
        self.grid_center = (self.grid_dim[0]//2, self.grid_dim[1]//2, (self.grid_dim[2]-1)//2)
        self.grid_length = 8.0
        self.grid_size = self.grid_length/self.grid_dim[0]
        self.grid_size_angle = 2*np.pi/(self.grid_dim[2]-1)    # if e.g. grid_dim[2] == 11, then 9 divisions

        self.pgo = PoseGraphOptimizerGTSAM()
        self.result = None
        self.marginals = None

        self.loop_closure_inserted = False

    def grab_frame(self, frame):
        self.current_frame = frame
        if not self.ref_keyframe:
            self.ref_keyframe = Keyframe(frame)
            self.ref_keyframe.key_point_initializer()
            self.ref_keyframe.set_pose(np.eye(3), [0, 0, 0])
            self.map.append(self.ref_keyframe)
            self.result, self.marginals = self.pgo.add_node_optimize(self.ref_keyframe)
            return True
        else:
            return self.track()

    def track(self):
        n_matched = 0
        candidate_ref_keyframe = None
        new_kf = None

        list_kf = [kf[0] for kf in self.ref_keyframe.neighbors]
        # if self.ref_keyframe not in list_kf:
        #     list_kf.append(self.ref_keyframe)
        if self.ref_keyframe in list_kf:
            list_kf.remove(self.ref_keyframe)
        list_kf = sorted(list_kf, key=lambda x: x.kfID, reverse=False)
        list_kf.append(self.ref_keyframe)

        list_kf_n = len(list_kf)
        list_kf_correspondence = []
        count_success = 0
        n_max = 0
        if list_kf_n == 1:
            self.insert_new_kf = True

        for i in range(list_kf_n - 1, -1, -1):
            key_fr = list_kf[i]

            if count_success <= 3 or (self.new_kf_inserted and not self.tracking_success):
                flag_success, rel_pose, matches = self.visual_odometry_teaser(self.current_frame, key_fr)
            else:
                break

            if not flag_success:
                del list_kf[i]      # is this really necessary?
                if i == list_kf_n-1:    # If tracking reference keyframe failed, insert new kf
                    self.insert_new_kf = True
                continue
            else:   # Decide if the current frame should be converted into a new keyframe
                list_kf_correspondence.insert(0, (rel_pose, matches))
                if i == list_kf_n-1 and (len(matches) < 0.55*key_fr.n_kp or len(matches) < 300):
                    self.insert_new_kf = True
                    print("Decided to convert to kf, matches: ", len(matches), " KPs: ", key_fr.n_kp, " KFID: ",
                          key_fr.kfID)
                if len(matches) > 0.85*key_fr.n_kp:
                    self.tracking_success = True
                    print("Tracking was successful!")
                # list_kf_correspondence[i] = len(matches)

            count_success += 1

            # if i == len(list_kf) - 1:
            rel_rot = rel_pose[:3, :3]
            rel_trans = rel_pose[:3, 3]
            rot = key_fr.rot().dot(rel_rot)  # rot = rel_rot.dot(key_fr.rot())
            trans = key_fr.pos() + key_fr.rot().dot(rel_trans)  # trans = rel_trans + rel_rot.dot(key_fr.pos())

            self.current_pose[0:3, 0:3] = rot
            self.current_pose[0:3, 3] = trans

            # if self.insert_new_kf and not self.new_kf_inserted:    # Convert current frame into new kf
            if self.insert_new_kf and not self.new_kf_inserted:  # Convert current frame into new kf
                self.new_kf_inserted = True
                new_kf = Keyframe(self.current_frame)       # Now add key points to the new kf
                self.map.append(new_kf)
                new_kf.set_pose(rot, trans)

            if self.new_kf_inserted:
                for p in matches:
                    idx_kf = p.queryIdx
                    idx_cf = p.trainIdx
                    kp = key_fr.get_key_point(idx_kf)
                    if kp:
                        new_kf.add_key_point(idx_cf, kp)
                    else:
                        d = key_fr.depth[idx_kf]
                        pos = np.array([(key_fr.fp[idx_kf].pt[0]-self.cx)/self.fx*d,
                                        (key_fr.fp[idx_kf].pt[1]-self.cy)/self.fy*d, d])
                        pos = key_fr.rot().dot(pos) + key_fr.pos()
                        kp = key_fr.new_key_point(idx_kf, pos)
                        new_kf.add_key_point(idx_cf, kp)
                print("New KF initialized: <", new_kf.kfID, "> ", len(new_kf.key_points))

                score = Keyframe.kfdb.score_l1(new_kf.bow, key_fr.bow, normalize=True)

                key_fr.neighbors.append((new_kf, rel_pose, score))
                new_kf.neighbors.append((key_fr, np.linalg.inv(rel_pose), score))

            # Change the reference kf to the one with max correspondence
            n_matched = len(matches)
            if n_matched > n_max:
                candidate_ref_keyframe = key_fr
                n_max = n_matched

        if self.new_kf_inserted:
            # self.ref_keyframe = new_kf
            print("New KF Neighbors: ", [kf[0].kfID for kf in new_kf.neighbors])
        if candidate_ref_keyframe:
            self.ref_keyframe = candidate_ref_keyframe
            print("REF KF: ", self.ref_keyframe.kfID, " keypoints: ", len(self.ref_keyframe.key_points),
                  " Neighbors: ", [kf_[0].kfID for kf_ in self.ref_keyframe.neighbors])
        else:
            print("Number of matched features: ", n_matched)

        # Check BOW vectors for loop closure detection
        if self.new_kf_inserted:
            list_candidates = Keyframe.kfdb.get_candidates(new_kf)
            print("THE LIST OF CANDIDATES FOR LOOP CLOSURE: ", [kf.kfID for kf in list_candidates])
            for kf in list_candidates:
                self.loop_closure_teaser(new_kf, kf)

        if self.new_kf_inserted:
            self.result, self.marginals = self.pgo.add_node_optimize(new_kf)

        self.insert_new_kf = False
        self.new_kf_inserted = False
        self.tracking_success = False

        return count_success > 0

    def visual_odometry_teaser(self, current_f, key_f):
        flag_reproject = True
        # kf_des = key_f.des
        # Fetch descriptors from the keypoints of key_frame
        kf_des = np.zeros([key_f.n_kp, 32], dtype=np.uint8)
        kf_kp_indices = []
        for idx, kp_idx in enumerate(key_f.key_points):
            kf_des[idx, :] = key_f.key_points[kp_idx].des
            kf_kp_indices.append(kp_idx)

        # Match keypoint descriptors with the features of the current frame
        matches = self.matcher.match(kf_des, current_f.des)
        # matches = sorted(matches, key=lambda x: x.distance)  # Sort them in the order of their distance.
        if len(matches) < 30:
            print("VO failed due to lack of feature matches: ", len(matches))
            return False, None, []
        # if len(matches) < key_f.n_kp * 0.55 or len(matches) < 300:
        #     self.insert_new_kf = True
        #     print("Decision to convert to kf, matches: ", len(matches), " KPs: ", key_f.n_kp, " KFID: ", key_f.kfID)

        src = np.zeros((3, len(matches)), dtype=float)
        dst = np.zeros((3, len(matches)), dtype=float)

        for idx, p in enumerate(matches):
            p.queryIdx = kf_kp_indices[p.queryIdx]
            src[:, idx] = self.obs_to_3d(current_f.fp[p.trainIdx].pt[0], current_f.fp[p.trainIdx].pt[1],
                                         current_f.depth[p.trainIdx])
            dst[:, idx] = self.obs_to_3d(key_f.fp[p.queryIdx].pt[0], key_f.fp[p.queryIdx].pt[1],
                                         key_f.depth[p.queryIdx])

        optim = PoseOptimizerTeaser()
        pose = optim.optimize(src, dst)
        rot = pose[0:3, 0:3]
        trans = pose[0:3, 3]

        edge_outlier = []
        for idx, p in enumerate(matches):
            pf = self.obs_to_3d(current_f.fp[p.trainIdx].pt[0], current_f.fp[p.trainIdx].pt[1],
                                current_f.depth[p.trainIdx])
            pkf = self.obs_to_3d(key_f.fp[p.queryIdx].pt[0], key_f.fp[p.queryIdx].pt[1], key_f.depth[p.queryIdx])
            error = pkf - rot.dot(pf) - trans
            # print(np.linalg.norm(error))
            if np.linalg.norm(error) < 2:
                edge_outlier.append(False)
            else:
                edge_outlier.append(True)

        if np.linalg.norm(pose[0:3, 3]) > 2:
            print("VO failed due to bad translation: ", np.linalg.norm(pose[0:3, 3]), " matches: ", len(matches))
            return False, None, []
        elif len(edge_outlier) - np.sum(edge_outlier) < 60:
            print("VO failed due to lack of enough matches: ", len(edge_outlier) - np.sum(edge_outlier))
            return False, None, []

        matches_inlier = [p for idx, p in enumerate(matches) if not edge_outlier[idx]]

        if flag_reproject:
            fp_inliers_idx_kf = [p.queryIdx for p in matches_inlier]
            fp_inliers_idx_f = [p.trainIdx for p in matches_inlier]
            new_matches = self.reproject_features(current_f, key_f, pose,
                                                  fp_inliers_idx_f, fp_inliers_idx_kf)
            matches_inlier.extend(new_matches)

        print("VO succeeded, init. inliers: ", len(edge_outlier) - np.sum(edge_outlier))
        return True, pose, matches_inlier

    def reproject_features(self, current_f, key_f, pose, fp_inliers_idx_f, fp_inliers_idx_kf):
        rot = pose[0:3, 0:3]
        trans = pose[0:3, 3]
        n_inliers = len(fp_inliers_idx_kf)
        # assert len(fp_inliers_idx_kf) == len(fp_inliers_idx_f)
        if len(key_f.fp)-n_inliers < 50 or len(current_f.fp)-n_inliers < 50:
            return []
        kf_des = np.empty([len(key_f.fp)-n_inliers, 32], dtype=np.uint8)
        f_des = np.empty([len(current_f.fp)-n_inliers, 32], dtype=np.uint8)
        kf_indices = []
        f_indices = []
        counter = 0
        for idx, fp in enumerate(current_f.fp):
            if idx in fp_inliers_idx_f:
                continue
            f_des[counter, :] = current_f.des[idx, :]
            counter += 1
            f_indices.append(idx)

        counter = 0
        for idx, fp in enumerate(key_f.fp):
            if idx in fp_inliers_idx_kf:
                continue
            kf_des[counter, :] = key_f.des[idx, :]
            counter += 1
            kf_indices.append(idx)

        matches = self.matcher.match(kf_des, f_des)

        n_reprojected = 0
        new_matches = []
        for p in matches:
            p.queryIdx = kf_indices[p.queryIdx]
            p.trainIdx = f_indices[p.trainIdx]
            dkf = key_f.depth[p.queryIdx]
            df = current_f.depth[p.trainIdx]
            pkf = self.obs_to_3d(key_f.fp[p.queryIdx].pt[0], key_f.fp[p.queryIdx].pt[1], dkf)
            pf = self.obs_to_3d(current_f.fp[p.trainIdx].pt[0], current_f.fp[p.trainIdx].pt[1]
                                , df)

            error = pkf - rot.dot(pf) - trans
            # print(np.linalg.norm(error))
            if np.linalg.norm(error) < self.reprojection_threshold:
                n_reprojected += 1
                kp = key_f.new_key_point(p.queryIdx, pkf)
                key_f.add_key_point(p.queryIdx, kp)

                new_matches.append(p)

        print(n_reprojected, " new keypoints created on kf ", key_f.kfID, "from tracking frame", current_f.id)
        return new_matches

    def obs_to_3d(self, u, v, d):
        return np.array([(u - self.cx) / self.fx * d, (v - self.cy) / self.fy * d, d])

    def obs_to_stereo(self, u, v, d):
        return np.array([u, u - self.bf / d, v])

    def loop_closure_teaser(self, new_kf, loop_kf):
        # If temporal gap is small, disregard this candidate
        if new_kf.kfID - loop_kf.kfID < 10:
            return
        # Calculate the similarity score and compare with neighbors
        new_score = Keyframe.kfdb.score_l1(new_kf.bow, loop_kf.bow, normalize=True)
        candidate_kf = loop_kf
        for kf, _, _ in loop_kf.neighbors:
            neighbor_score = Keyframe.kfdb.score_l1(new_kf.bow, kf.bow, normalize=True)
            if neighbor_score > new_score:
                new_score = neighbor_score
                candidate_kf = kf
        loop_kf = candidate_kf

        if loop_kf in new_kf.neighbors_list():
            return

        min_score = 1
        for _, _, score in loop_kf.neighbors:
            min_score = min(min_score, score)

        if min_score > new_score:
            return
        # If the absolute positions and orientations are not close enough, return
        # if np.linalg.norm(new_kf.pos()-loop_kf.pos()) > 2 or \
        #         rot_to_angle(np.matmul(new_kf.rot(), loop_kf.rot().T)) > 20/180*np.pi*np.sqrt(2):
        #     return

        # Find matches, and return if few matches
        matches = self.matcher.match(loop_kf.des, new_kf.des)
        if len(matches) < 100:
            return

        src = np.zeros((3, len(matches)), dtype=float)
        dst = np.zeros((3, len(matches)), dtype=float)
        for idx, p in enumerate(matches):
            src[:, idx] = self.obs_to_3d(new_kf.fp[p.trainIdx].pt[0], new_kf.fp[p.trainIdx].pt[1],
                                         new_kf.depth[p.trainIdx])
            dst[:, idx] = self.obs_to_3d(loop_kf.fp[p.queryIdx].pt[0], loop_kf.fp[p.queryIdx].pt[1],
                                         loop_kf.depth[p.queryIdx])
        optim = PoseOptimizerTeaser()
        pose = optim.optimize(src, dst)
        rot = pose[0:3, 0:3]
        trans = pose[0:3, 3]

        errors = dst-rot.dot(src)-trans.reshape((3, 1))
        outliers = []
        n_outliers = 0
        n_inliers = 0
        for idx in range(len(matches)):
            if np.linalg.norm(errors[:, idx]) < self.reprojection_threshold:
                outliers.append(False)
                n_inliers += 1
            else:
                outliers.append(True)
                n_outliers += 1

        if n_inliers < 150:  # n_outliers > 0.65 * len(matches) or
            return

        odom_rel_rot = np.matmul(loop_kf.rot().T, new_kf.rot())
        odom_rel_pos = np.matmul(loop_kf.rot().T, new_kf.pos()-loop_kf.pos())
        if np.linalg.norm(pose[0:3, 3]-odom_rel_pos) > 0.5 or \
                rot_to_angle(odom_rel_rot.T.dot(pose[0:3, 0:3])) > 10/180*np.pi:
            return

        loop_kf.neighbors.append((new_kf, pose, new_score))
        new_kf.neighbors.append((loop_kf, np.linalg.inv(pose), new_score))
        self.n_loop_closures += 1
        print("LOOP CLOSURE ESTABLISHED WITH KF: ", loop_kf.kfID)
        self.loop_closure_inserted = True
        # Need to merge keypoints!

    def get_state(self, pcd):
        state = np.zeros(self.grid_dim, dtype=np.float32)
        # state_xz = np.zeros((self.grid_dim[0], self.grid_dim[1]), dtype=np.float32)
        # state_h = np.zeros(self.grid_dim[2], dtype=np.float32)

        current_rot = self.current_pose[0:3, 0:3]
        current_pos = self.current_pose[0:3, 3]

        kf_center_list = []

        n_good_kf = 0
        for kf in self.map:
            rel_rot = current_rot.T.dot(kf.rot())
            rel_pos = current_rot.T.dot(kf.pos()-current_pos)

            add_ = 0
            if self.grid_dim[0] % 2 == 1:
                add_ = 0.5*self.grid_size
            x_idx = int((rel_pos[0]+add_)//self.grid_size+self.grid_center[0])
            z_idx = int((rel_pos[2]+add_)//self.grid_size+self.grid_center[1])

            # Only consider keyframes that are either in the grid or one idx away in both directions
            if x_idx < -1 or x_idx > self.grid_dim[0]:
                continue
            if z_idx < -1 or z_idx > self.grid_dim[1]:
                continue

            if 0 <= x_idx < self.grid_dim[0] and 0 <= z_idx < self.grid_dim[1]:
                kf_center_list.append((x_idx, z_idx))

            n_good_kf += 1

            # heading = rot_to_heading(rel_rot)
            # heading_idx = int((np.pi + heading) // self.grid_size_angle)

            # cov = self.marginals.marginalCovariance(gt.symbol_shorthand_X(kf.kfID))
            inf = self.marginals.marginalInformation(X(kf.kfID))
            inf[0:3, 0:3] = inf[0:3, 0:3].dot(rel_rot.T)
            inf[3:, 3:] = inf[3:, 3:].dot(rel_rot.T)
            rel_cov_pos = 0
            rel_cov_rot = 0

            for x_ in range(self.grid_dim[0]):
                for z_ in range(self.grid_dim[1]):
                    p_ = np.array([(x_-self.grid_center[0])*self.grid_size,
                                   0,
                                   (z_-self.grid_center[1])*self.grid_size])
                    p_ = rel_pos - p_
                    prob = np.exp(-0.5*p_.reshape((1, 3)).dot(inf[0:3, 0:3]).dot(p_.reshape((3, 1))))
                    prob = prob*np.sqrt(np.linalg.det(inf[0:3, 0:3]))
                    state[x_, z_, 0:self.grid_dim[2]-1] += min(prob.squeeze(), 2)
                    # state_xz[x_, z_] += prob.squeeze()
            for h_ in range(self.grid_dim[2]-2):
                theta = (h_ - self.grid_center[2]) * self.grid_size_angle
                rot = np.array([[np.cos(theta), 0, -np.sin(theta)],
                                [0            , 1,       0       ],
                                [np.sin(theta), 0, np.cos(theta) ]])
                rot = rel_rot.T.dot(rot)
                r_ = Rotation.from_matrix(rot).as_rotvec()
                prob = np.exp(-0.5 * r_.reshape((1, 3)).dot(inf[3:, 3:]).dot(r_.reshape((3, 1))))
                prob = prob * np.sqrt(np.linalg.det(inf[3:, 3:]))
                state[:, :, h_] += min(prob.squeeze(), 2)
                # state_h[h_] += prob.squeeze()

            # state[x_idx, z_idx, 2*heading_idx] += np.linalg.det(cov[0:3, 0:3])
            # state[x_idx, z_idx, 2 * heading_idx+1] += np.linalg.det(cov[3:, 3:])

        # average over the qualified keyframes
        if n_good_kf > 0:
            state = state / n_good_kf

        points = np.array(pcd.points)
        points = np.matmul(points - current_pos.reshape(1, 3), current_rot)
        valid_idx = np.where(np.linalg.norm(points[:, 0:3:2], ord=np.inf, axis=1) < self.grid_length/2)
        points = points[valid_idx[0], :]
        valid_idx2 = np.where(np.abs(points[:, 1]) < 1.5)
        points = points[valid_idx2[0], :]
        valid_idx3 = np.where(np.linalg.norm(points[:, 0:3:2], ord=np.inf, axis=1) != 0.)
        points = points[valid_idx3[0], :]

        # if points.size != 0:
        #     colors = np.array(pcd.colors)
        #     colors = colors[valid_idx[0], :]
        #     colors = colors[valid_idx2[0], :]
        #     pcd_ = o3d.geometry.PointCloud()
        #     pcd_.points = o3d.utility.Vector3dVector(points)
        #     pcd_.colors = o3d.utility.Vector3dVector(colors)
        #     o3d.visualization.draw_geometries([pcd_])

        for idx in range(points.shape[0]):
            x_idx = int((points[idx, 0] + add_) // self.grid_size + self.grid_dim[0] // 2)
            z_idx = int((points[idx, 2] + add_) // self.grid_size + self.grid_dim[1] // 2)
            state[x_idx, z_idx, -1] = 1.
        # state[:, :, -1] = state[:, :, -1] / np.max(state[:, :, -1])
        for x_idx, z_idx in kf_center_list:
            state[x_idx, z_idx, -1] = 0.
        return state

    def check_loop_closure(self):
        temp = self.loop_closure_inserted
        self.loop_closure_inserted = False
        return temp

    def reset(self):
        Keyframe.kfID = 0
        Keyframe.kfdb = KeyframeDatabase()
        Frame.id = 0
        self.current_frame = None
        self.ref_keyframe = None
        self.image_queue = []  # ?

        self.insert_new_kf = False
        self.new_kf_inserted = False
        self.tracking_success = False

        self.map = []
        # self.current_rot = np.eye(3)
        # self.current_pos = np.array([0, 0, 0])
        self.current_pose = np.eye(4)

        self.n_loop_closures = 0  # Counter for the number of verified loop closures

        self.pgo = PoseGraphOptimizerGTSAM()
        self.result = None
        self.marginals = None

        self.loop_closure_inserted = False

