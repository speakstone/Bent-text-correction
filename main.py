# -*- coding: UTF-8 -*-

import numpy as np
import cv2
from cfg import *

img = cv2.imread('image.png')

c_src = np.array([
    [0.0, 0.0],
    [1., 0],
    [1, 1],
    [0, 1],
    [0.3, 0.3],
    [0.7, 0.7],
])

c_dst = np.array([
    [0., 0],
    [1., 0],
    [1, 1],
    [0, 1],
    [0.4, 0.4],
    [0.6, 0.6],
])

warped = warp_image_cv(img, c_src, c_dst, dshape=(512, 512))
show_warped(img, c_src, c_dst, warped)
