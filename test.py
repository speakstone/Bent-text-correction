# -*- coding: UTF-8 -*-
import cv2
import numpy as np

from cfg import *

ix, iy = -1, -1 # 鼠标左键按下时的坐标
ix_1, iy_1 = None, None # 前一个坐标点
draw_label = []
draw_label_i = []

def draw_circle(event, x, y, flags, param):
    global ix, iy, ix_1, iy_1, draw_label_i ,draw_label

    if event == cv2.EVENT_LBUTTONDOWN:
        # 鼠标左键按下事件
        ix, iy = x, y
        draw_label_i.append([ix, iy])
        print("ix, iy", ix, iy)
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        if ix_1 is not None:
            cv2.line(img, (ix_1, iy_1), (ix, iy), (255, 0, 0), 1, 4)
        ix_1, iy_1 = x, y

    elif event == cv2.EVENT_RBUTTONDOWN:
        # 鼠标右键按下事件
        draw_label.append(draw_label_i)
        draw_label_i = draw_label_i[-1]

# img = np.zeros((512, 512, 3), np.uint8)
img = cv2.imread("seal.jpg")
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle) #设置鼠标事件的回调函数

while(1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(' '):
        print(draw_label)
        w, h, _ = img.shape
        print("w, h", w, h)
        c_src = original_coordinates_get(draw_label, w, h)
        c_dst = transfor_coordinates_get(draw_label, (w, h))
        warped = warp_image_cv(img, c_src, c_dst, dshape=(512, 512))
        show_warped(img, c_src, c_dst, warped)
        break



# cv2.destroyAllWindows()