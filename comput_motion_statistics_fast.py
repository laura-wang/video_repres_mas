import numpy as np



bin_size = 8
angle_unit = 360 / bin_size


def cell_gradient(cell_magnitude, cell_angle):
    orientation_centers = [0] * bin_size
    for k in range(cell_magnitude.shape[0]):
        for l in range(cell_magnitude.shape[1]):
            gradient_strength = cell_magnitude[k][l]
            gradient_angle = cell_angle[k][l]
            min_angle = int(gradient_angle / angle_unit) % 8
            max_angle = (min_angle + 1) % bin_size
            mod = gradient_angle % angle_unit
            orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
            orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))
    return orientation_centers



def pattern_1(mag,ang):

    max_sum = 0
    max_idx = []
    for i in range(4):
        for j in range(4):
            x_start = i * 28
            x_end = x_start + 28
            y_start = j * 28
            y_end = y_start + 28

            tmp_block = mag[x_start:x_end, y_start:y_end]
            # print(tmp_block.shape)

            block_sum = np.sum(tmp_block)
            if block_sum > max_sum:
                max_sum = block_sum
                max_block = 4 * i + j + 1

    j = (max_block - 1) % 4
    i = int((max_block - 1 - j) / 4)

    x_start = i * 28
    x_end = x_start + 28
    y_start = j * 28
    y_end = y_start + 28

    max_mag_block = mag[x_start:x_end, y_start:y_end]
    max_ang_block = ang[x_start:x_end, y_start:y_end]


    orientation_center = cell_gradient(max_mag_block, max_ang_block)

    max_idx = np.argmax(orientation_center) + 1

    return max_block, max_idx



def pattern_2(mag,ang):
    total_sum = []
    block1 = mag[42:70, 42:70] # 28 x 28
    sum_1 = np.sum(block1)
    block1_sum = sum_1 / (28 * 28)
    total_sum.append(block1_sum)


    block2 = mag[28:84, 28:84] # 56 * 56
    sum_2 = np.sum(block2)
    block2_sum = (sum_2 - sum_1) / (56 * 56 - 28 * 28)
    total_sum.append(block2_sum)

    block3 = mag[14:98, 14:98] # 84 * 84
    sum_3 = np.sum(block3)
    block3_sum = (sum_3 - sum_2) / (84* 84- 56 * 56)
    total_sum.append(block3_sum)


    block4 = mag[0:112, 0:112] # 112 x 112
    sum_4 = np.sum(block4)
    block4_sum = (sum_4 - sum_3) / (112 * 112 - 84 * 84)
    total_sum.append(block4_sum)

    max_idx = total_sum.index(max(total_sum))

    if max_idx == 0:  # block 1 28 x 28
        max_mag_block = mag[42:70, 42:70]
        max_ang_block = ang[42:70, 42:70]
    elif max_idx == 1:  # block 2 56 x 56
        tmp_mag = np.zeros_like(block2)
        tmp_mag[14:42, 14:42] = mag[42:70, 42:70]
        max_mag_block = mag[28:84, 28:84] - tmp_mag

        tmp_ang = np.zeros_like(block2)
        tmp_ang[14:42, 14:42] = ang[42:70, 42:70]
        max_ang_block = ang[28:84, 28:84] - tmp_ang

    elif max_idx == 2:  # block 3 84 x 84
        tmp_mag = np.zeros_like(block3)
        tmp_mag[14:70, 14:70] = mag[28:84, 28:84]
        max_mag_block = mag[14:98, 14:98] - tmp_mag

        tmp_ang = np.zeros_like(block3)
        tmp_ang[14:70, 14:70] = ang[28:84, 28:84]
        max_ang_block = ang[14:98, 14:98] - tmp_ang

    elif max_idx == 3:  # block 4 112 x 112
        tmp_mag = np.zeros_like(block4)
        tmp_mag[14:98, 14:98] = mag[14:98, 14:98]
        max_mag_block = mag[0:112, 0:112] - tmp_mag

        tmp_ang = np.zeros_like(block4)
        tmp_ang[14:98, 14:98] = ang[14:98, 14:98]
        max_ang_block = ang[0:112, 0:112] - tmp_ang


    orientation_center = cell_gradient(max_mag_block, max_ang_block)

    max_ang = np.argmax(orientation_center)


    return (max_idx+1), (max_ang+1)



def pattern_3(mag,ang):

    mag_block_all = []
    ang_block_all = []

    block_one = mag[0:56,0:56]
    mag_block_1 = block_one[np.tril_indices(56)]
    mag_block_2 = block_one[np.triu_indices(56)]

    ang_block_one = ang[0:56, 0:56]
    ang_block_1 = ang_block_one[np.tril_indices(56)]
    ang_block_2 = ang_block_one[np.triu_indices(56)]

    ############################################################

    block_two = mag[0:56, 56:112]
    block_two = np.flip(block_two,1)

    mag_block_3 = block_two[np.triu_indices(56)]
    mag_block_4 = block_two[np.tril_indices(56)]


    ang_block_two = ang[0:56, 56:112]
    ang_block_two = np.flip(ang_block_two, 1)
    ang_block_3 = ang_block_two[np.triu_indices(56)]
    ang_block_4 = ang_block_two[np.tril_indices(56)]


    ################################################
    block_three = mag[56:112, 0:56]
    block_three = np.flip(block_three,1)
    mag_block_5 = block_three[np.triu_indices(56)]
    mag_block_6 = block_three[np.tril_indices(56)]


    ang_block_three = ang[56:112, 0:56]
    ang_block_three = np.flip(ang_block_three, 1)
    ang_block_5 = ang_block_three[np.triu_indices(56)]
    ang_block_6 = ang_block_three[np.tril_indices(56)]

    ############################################
    block_four = mag[56:112, 56:112]
    mag_block_7 = block_four[np.tril_indices(56)]
    mag_block_8 = block_four[np.triu_indices(56)]


    ang_block_four = ang[56:112, 56:112]
    ang_block_7 = ang_block_four[np.tril_indices(56)]
    ang_block_8 = ang_block_four[np.triu_indices(56)]

    ############################################


    mag_block_all.append(mag_block_1)
    mag_block_all.append(mag_block_2)
    mag_block_all.append(mag_block_3)
    mag_block_all.append(mag_block_4)
    mag_block_all.append(mag_block_5)
    mag_block_all.append(mag_block_6)
    mag_block_all.append(mag_block_7)
    mag_block_all.append(mag_block_8)
    mag_block_all = np.array(mag_block_all)
    sum_all = np.sum(mag_block_all,1)

    ang_block_all.append(ang_block_1)
    ang_block_all.append(ang_block_2)
    ang_block_all.append(ang_block_3)
    ang_block_all.append(ang_block_4)
    ang_block_all.append(ang_block_5)
    ang_block_all.append(ang_block_6)
    ang_block_all.append(ang_block_7)
    ang_block_all.append(ang_block_8)
    ang_block_all = np.array(ang_block_all)


    #############################################
    max_mag_idx = np.argmax(sum_all)

    max_mag_block = mag_block_all[max_mag_idx]
    max_ang_block = ang_block_all[max_mag_idx]

    max_mag_block = np.reshape(max_mag_block, [12, 133])
    max_ang_block = np.reshape(max_ang_block, [12, 133])

    orientation_center = cell_gradient(max_mag_block, max_ang_block)
    max_ang_idx = np.argmax(orientation_center)


    return max_mag_idx+1, max_ang_idx+1




