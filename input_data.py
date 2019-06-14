from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2


from scipy import ndimage
import multiprocessing
from comput_motion_statistics_fast import pattern_1, pattern_2, pattern_3


new_height = 128
new_width = 171
crop_size= 112
clip_length=16




def read_batch(input):

    rgb_line,u_flow_line,v_flow_line = input




    rgb_img_dir = rgb_line[0]
    start_frame = int(rgb_line[1])


    crop_x = random.randint(0, new_height - crop_size)  # crop size should be applied to all images
    crop_y = random.randint(0, new_width - crop_size)
    clip_sample_one = []
    for i in range(clip_length):
        cur_img_path = os.path.join(rgb_img_dir, "frame" + "{:06}.jpg".format(start_frame + i))

        img_origin = cv2.imread(cur_img_path)

        img_res = cv2.resize(img_origin, (171, 128))
        img = img_res.astype(np.float32)

        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]

        clip_sample_one.append(img)

    clip_sample_one = np.array(clip_sample_one).astype(np.float32)  # 16 x 112 x 112 x 3

    u_flow_dir = u_flow_line[0]

    v_flow_dir = v_flow_line[0]

    label_sample_one = []
    # compute sum of motion boundaries on u_flow
    du_x = 0
    du_y = 0

    du_x_all = []
    du_y_all = []

    for i in range(clip_length - 1):
        cur_img_path = os.path.join(u_flow_dir, 'frame' + '{:06}.jpg'.format(start_frame + i))
        img = cv2.imread(cur_img_path)
        img = img[..., 0]
        img = cv2.resize(img, (171, 128))

        u_flow = img.astype(np.float32)
        u_flow = ((u_flow * 40.) / 255.) - 20


        u_flow = u_flow[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size]


        mx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        my = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        d_x = ndimage.convolve(u_flow, mx)
        d_y = ndimage.convolve(u_flow, my)

        du_x_all.append(d_x)
        du_y_all.append(d_y)

        du_x += d_x
        du_y += d_y

    # compute sum of motion boundaries on v_flow
    dv_x = 0
    dv_y = 0

    dv_x_all = []
    dv_y_all = []

    for i in range(clip_length - 1):
        cur_img_path = os.path.join(v_flow_dir, 'frame' + '{:06}.jpg'.format(start_frame + i))
        img = cv2.imread(cur_img_path)
        img = img[..., 0]
        img = cv2.resize(img, (171, 128))

        v_flow = img.astype(np.float32)
        v_flow = ((v_flow * 40.) / 255.) - 20


        v_flow = v_flow[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size]



        mx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        my = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        d_x = ndimage.convolve(v_flow, mx)
        d_y = ndimage.convolve(v_flow, my)

        dv_x_all.append(d_x)
        dv_y_all.append(d_y)

        dv_x += d_x
        dv_y += d_y

    mag_u, ang_u = cv2.cartToPolar(du_x, du_y, angleInDegrees=True)
    mag_v, ang_v = cv2.cartToPolar(dv_x, dv_y, angleInDegrees=True)


    u_max_mag_1, u_max_ang_1 = pattern_1(mag_u, ang_u)
    v_max_mag_1, v_max_ang_1 = pattern_1(mag_v, ang_v)

    u_max_mag_2, u_max_ang_2 = pattern_2(mag_u, ang_u)
    v_max_mag_2, v_max_ang_2 = pattern_2(mag_v, ang_v)

    u_max_mag_3, u_max_ang_3 = pattern_3(mag_u, ang_u)
    v_max_mag_3, v_max_ang_3 = pattern_3(mag_v, ang_v)




    ### compute max du_all

    du_sum_mag =[]
    for i in range(15):
        cur_du_x = du_x_all[i]
        cur_du_y = du_y_all[i]
        mag_u, ang_u = cv2.cartToPolar(cur_du_x, cur_du_y, angleInDegrees=True)
        tmp_sum_mag = np.sum(mag_u)
        du_sum_mag.append(tmp_sum_mag)

    du_sum_mag = np.array(du_sum_mag)
    max_du_idx = np.argmax(du_sum_mag) + 1  # start from 1


    ### compute max dv_all

    dv_sum_mag = []

    for i in range(15):
        cur_dv_x = dv_x_all[i]
        cur_dv_y = dv_y_all[i]

        mag_v, ang_v = cv2.cartToPolar(cur_dv_x, cur_dv_y, angleInDegrees=True)
        tmp_sum_mag = np.sum(mag_v)
        dv_sum_mag.append(tmp_sum_mag)

    dv_sum_mag = np.array(dv_sum_mag)
    max_dv_idx = np.argmax(dv_sum_mag) + 1  # start from 1




    label_sample_one.append(u_max_mag_1)
    label_sample_one.append(u_max_ang_1)
    label_sample_one.append(v_max_mag_1)
    label_sample_one.append(v_max_ang_1)

    label_sample_one.append(u_max_mag_2)
    label_sample_one.append(u_max_ang_2)
    label_sample_one.append(v_max_mag_2)
    label_sample_one.append(v_max_ang_2)

    label_sample_one.append(u_max_mag_3)
    label_sample_one.append(u_max_ang_3)
    label_sample_one.append(v_max_mag_3)
    label_sample_one.append(v_max_ang_3)

    label_sample_one.append(max_du_idx)
    label_sample_one.append(max_dv_idx)

    label_sample_one = np.array(label_sample_one)



    return clip_sample_one,label_sample_one


def read_all(rgb_filename, u_flow_filename, v_flow_filename, batch_size, start_pos=-1,shuffle=True, cpu_num=1):
    rgb_lines = open(rgb_filename, 'r')
    rgb_lines = list(rgb_lines)

    u_flow_lines = open(u_flow_filename, 'r')
    u_flow_lines = list(u_flow_lines)

    v_flow_lines = open(v_flow_filename, 'r')
    v_flow_lines = list(v_flow_lines)

    batch_index = 0
    next_batch_start = -1

    train_clips = []
    label = []


    if start_pos < 0:
        shuffle=True

    if shuffle:
        video_indices = list(range(len(rgb_lines)))
        random.shuffle(video_indices)  # shuffle index!
    else:
        video_indices = range(start_pos, len(rgb_lines))

    lines_batch = []
    for index in video_indices:

        if (batch_index >= batch_size):  # get 30 samples
            next_batch_start = index
            break
        else:
            rgb_line = rgb_lines[index].strip('\n').split()
            #print(rgb_line)
            u_flow_line = u_flow_lines[index].strip('\n').split()
            v_flow_line = v_flow_lines[index].strip('\n').split()
            lines_batch.append((rgb_line,u_flow_line,v_flow_line))
            batch_index = batch_index + 1

    data = (lines_batch)
    p = multiprocessing.Pool(processes=cpu_num)

    results = p.map(read_batch, data)  # results: 16 x 8

    p.close()
    p.join()

    train_clips = []
    label = []
    for result in results:  # 30 x 16 x 112 x 112 x 3 label: 30 x 1
        sample_one, label_one = result
        train_clips.append(sample_one)
        label.append(label_one)

    np_train_clips = np.array(train_clips).astype(np.float32)  # N x 16 x 112 x 112 x 3
    np_arr_label = np.array(label).astype(np.float32)

    return np_train_clips, np_arr_label, next_batch_start








