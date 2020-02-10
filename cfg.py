import cv2
import matplotlib.pyplot as plt
import tps as tps
import numpy as np

def show_warped(img, c_src, c_dst, warped):
    fig, axs = plt.subplots(1, 2, figsize=(16,8))
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].imshow(img[...,::-1], origin='upper')
    axs[0].scatter(c_src[:, 0]*img.shape[1], c_src[:, 1]*img.shape[0], marker='+', color='black')
    axs[1].imshow(warped[...,::-1], origin='upper')
    axs[1].scatter(c_dst[:, 0]*warped.shape[1], c_dst[:, 1]*warped.shape[0], marker='+', color='black')
    plt.show()

def warp_image_cv(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

def original_coordinates_get(draw_label, w, h ):
    c_src = []
    for draw_i in draw_label:
        for draw_j in draw_i:
            c_src.append(draw_j)
    c_src = np.array(c_src,  astype=np.float32)
    c_src /= [w, h]
    return c_src

def transfor_coordinates_get(draw_label, w, h):
    d_src = []
    x_y = np.zeros(2)
    for i in range(2):
        draw_i = np.array(draw_label[i],  astype=np.float32)
        for j in range(len(draw_i)-1):
            x_y[i] += np.sqrt(np.sum(np.square(draw_i[j+1]-draw_i[j])))
    [x_0 , y_0] = [w/10.0, h/5.0]
    for i in range(len(draw_label)):
        i_shape = len(draw_label[i])
        i_label = np.zeros([i_shape, 2]) + [x_0, y_0]
        i_add = np.array(range(0,  int(x_y[int(i % 2)] / i_shape) * i_shape, int(x_y[int(i % 2)] / i_shape)))
        i_label  = i_label + i_add