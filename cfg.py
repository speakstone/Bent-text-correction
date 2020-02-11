import cv2
import matplotlib.pyplot as plt
import tps as tps
import numpy as np

def image_slice(img, c_src):
    h, w, _ = img.shape
    c_src = c_src * [w,h]
    c_src = c_src.swapaxes(0, 1)
    x1, y1 = min(c_src[0]), min(c_src[1])
    x2, y2 = max(c_src[0]), max(c_src[1])
    img_r = img[int(y1*0.9):int(y2*1.1), int(x1*0.9):int(x2*1.1)]
    return img_r


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
    c_result = []
    for draw_i in draw_label:
        for draw_j in draw_i:
            c_src.append(draw_j)
    for i_src in c_src:
        if i_src not in c_result:
            c_result.append(i_src)
    # c_src = np.unique(np.array(c_src, dtype=np.float32), axis=0)
    c_result = np.array(c_result, dtype=np.float32)
    c_result /= [w, h]
    return c_result

def transfor_coordinates_get(draw_label, w, h):
    d_src = []
    d_result = []
    d_add = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
    x_y = np.zeros(2)
    for i in range(2):
        draw_i = np.array(draw_label[i], dtype=np.float32)
        for j in range(len(draw_i)-1):
            x_y[i] += np.sqrt(np.sum(np.square(draw_i[j+1]-draw_i[j])))
    # [x_0, y_0] = draw_label[0][0]
    [x_0, y_0] = [w/10 , h/2]
    for i in range(len(draw_label)):
        i_len = len(draw_label[i])
        i_width = x_y[i % 2]
        i_label = np.ones([i_len, 2])*d_add[i]
        i_add = np.array(range(0, int(i_width/i_len)*i_len, int(i_width/i_len))).reshape([-1, 1])
        i_src = i_label * i_add + [x_0, y_0]
        [d_src.append(x) for x in i_src]
        [x_0, y_0] = i_src[-1]

    for i_src in d_src:
        i_src = list(i_src)
        if i_src in d_result:
            continue
        else:
            d_result.append(i_src)
    # c_src = np.unique(np.array(c_src, dtype=np.float32), axis=0)
    d_result = np.array(d_result, dtype=np.float32)
    d_result /= [w, h]
    return d_result

def transfor_coordinates_get_x(draw_label, w, h):
    d_src = []
    d_result = []
    d_add = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])

    [x_0, y_0] = [w/10 , h/2]
    for x_i in range(len(draw_label)):
        x_draw = np.array(draw_label[x_i])
        x_y_start = [x_0, y_0]
        d_src.append(x_y_start)
        for y_j in range(1, len(x_draw)):
            x_len = np.sqrt(np.sum(np.square(x_draw[y_j] - x_draw[0])))
            x_y_j = x_y_start + d_add[x_i]*x_len
            d_src.append(x_y_j)
        x_y_start = list(x_y_j)

    for i_src in d_src:
        i_src = list(i_src)
        if i_src in d_result:
            continue
        else:
            d_result.append(i_src)
    d_result = np.array(d_result, dtype=np.float32)
    d_result /= [w, h]
    return d_result


def transfor_coordinates_get_x_y(draw_label, w, h):
    d_src = []
    x_label = []
    x_y_start_label = [[w / 10, h / 10]]
    [x_label.append(x[0]) for x in draw_label]
    x_label = np.array(x_label)
    for i in range(1, len(x_label)):
        x_wight = np.sqrt(np.sum(np.square(x_label[i]-x_label[0])))
        x_y_start_label.append([w / 10, h / 10 + x_wight])
    x_y_start_label = np.array(x_y_start_label, dtype = np.float32)
    for x_i in range(len(draw_label)):
        x_draw = np.array(draw_label[x_i])
        x_y_start = x_y_start_label[x_i]
        d_src.append(x_y_start)
        for y_j in range(1, len(x_draw)):
            x_len = np.array([np.sqrt(np.sum(np.square(x_draw[y_j] - x_draw[0]))), 0])
            x_y_j = x_y_start + x_len
            d_src.append(x_y_j)
    d_src = np.array(d_src, dtype=np.float32)
    d_src /= [w, h]
    return d_src

w, h  = 651, 650
a = np.array([[0.09999999,0.5], [0.09999999,0.47846153], [0.09999999,0.45692307], [0.11996928,0.45692307],
 [0.13993855,0.45692307], [0.15990783,0.45692307], [0.15990783,0.47846153], [0.15990783,0.5],
 [0.13379416,0.5], [0.10768049,0.5]])
img = cv2.imread("seal.jpg")
images = image_slice(img, a)

