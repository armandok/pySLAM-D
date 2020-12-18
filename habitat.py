import habitat_sim
import open3d as o3d

import random
import matplotlib.pyplot as plt

import numpy as np

from PIL import Image
import os


from Tracking import *
from Frame import *
from Optimizer import *

import gtsam as gt

from Simulator import Simulator


def kb_input():
    while not kb.kbhit():
        continue
    key_ = kb.getch()
    return key_


def pc_input():
    key_ = input()
    if len(key_) > 1:
        key_ = key_[0]
    return key_


def rand_input():
    key_ = np.random.choice(['w', 'a', 'd'])
    return key_


simulator = Simulator()
system = Tracking()

flag_random = False
flag_kbhit = False
plt.figure()

if flag_random:
    input_func = rand_input
elif flag_kbhit:
    kb = KBHit()
    input_func = kb_input
else:
    input_func = pc_input

kf_counter = 0

pcd = o3d.geometry.PointCloud()
config = Config()
save_dir = config.dir
while True:
    key = input_func()

    if len(key) == 0:
        continue

    if key == 'w':
        action = 'move_forward'
    elif key == 's':
        action = 'move_backward'
    elif key == 'a':
        action = 'turn_left'
    elif key == 'd':
        action = 'turn_right'
    elif ord(key) == 27 or key == 'c':
        break
    else:
        continue
    print("action", action)
    plt.clf()
    rgb, depth = simulator.get_obs(action)

    frame = Frame(np.array(rgb), np.array(depth), 0, bgr=False, add_noise=True)
    flag = system.grab_frame(frame)
    if not flag:
        print("FAILED!")

    # plt.imshow(rgb_img)
    plt.imshow(frame.draw_points(rgb))
    plt.draw()
    plt.pause(0.001)

    if kf_counter < len(system.map):    # This means that we have a new keyframe
        new_kf = system.map[-1]
        frame.save(rgb, depth, save_dir, str(new_kf.kfID))
        # print("KF: ", new_kf.kfID, " ", new_kf.pose_matrix())
        new_pcd, _ = frame.get_point_cloud(rgb[:, :, :3], depth, flag_rgb=True)
        new_pcd.transform(new_kf.pose_matrix())
        pcd += new_pcd
    
        kf_counter += 1


if flag_kbhit:
    kb.set_normal_term()

# pg_optimizer = PoseGraphOptimizerGTSAM(system.map)
# result, marginals = pg_optimizer.optimize()
result, marginals = system.result, system.marginals
# print(result)

list_images = [o.split(".")[0] for o in os.listdir(save_dir) if o[-3:] == "png"]
list_images.sort(key=int)
for name in list_images:
    img = Image.open(os.path.join(save_dir, name+".png"))
    im = np.array(img)
    depth = np.load(os.path.join(save_dir, name+".npy"))
    new_pcd, _ = Frame.get_point_cloud(im[:, :, :3], depth, flag_rgb=True)
    # new_pcd.transform(system.map[int(name)].pose_matrix())
    new_pcd.transform(result.atPose3(gt.symbol_shorthand_X(int(name))).matrix())
    cov = marginals.marginalCovariance(gt.symbol_shorthand_X(int(name)))
    print(name, ", Cov: ", np.linalg.norm(cov), " , Pos:", system.map[int(name)].pos())
    pcd += new_pcd
    pcd = pcd.voxel_down_sample(voxel_size=0.1)
    print("Number of points: ", len(pcd.points))
o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud(os.path.join(save_dir, "pcd.pcd"), pcd)
