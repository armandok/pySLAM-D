import numpy as np


def rot_to_angle(rot):
    return np.arccos(0.5*np.trace(rot)-0.5)


def rot_to_heading(rot):
    # This function calculates the heading angle of the rot matrix w.r.t. the y-axis
    new_rot = rot[0:3:2, 0:3:2]  # remove the mid row and column corresponding to the y-axis
    new_rot = new_rot/np.linalg.det(new_rot)
    return np.arctan2(new_rot[1, 0], new_rot[0, 0])
