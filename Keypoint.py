import cv2
import numpy as np
from Config import Config

import sys
# sys.path.append(r'~/Code/pyfbow/build/pyfbow.so')
import pyfbow


class KeyPoint:
    id = 0

    def __init__(self, pos, des):

        self.pos = pos
        self.des = des

        self.id = id

        KeyPoint.id += 1


    def get_id(self):
        return self.id
