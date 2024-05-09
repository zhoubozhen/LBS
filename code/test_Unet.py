# Created Time    : 2024/1/5 15:35
# Author  :  Bozhen Zhou (zbz22@mails.tsinghua.edu.cn)
# Tips    : Make sure the paths are correct. Change the output path if necessary.

# test program: get the corresponding prediction images from the trained model
from keras.models import load_model
import os
import numpy as np
import cv2


def sort_str(str):
    if str:
        try:
            holo_count = str.split('_')[0][2:]
        except:
            holo_count = -1
    return int(holo_count)


def get_data_test():
    path_in = r"../data/experiment_data/test_in"
    t_in = []
    img_in_list = os.listdir(path_in)
    # img_in_list_new = sorted(img_in_list, key=sort_str)
    img_in_list_new = img_in_list
    for name in img_in_list_new:
        img_in = cv2.imread(path_in + '/' + name, cv2.IMREAD_GRAYSCALE)
        img_in = np.resize(img_in, (256, 256, 1))
        t_in.append(img_in)
    return np.array(t_in), img_in_list_new


model = load_model('trained_model.h5')
(test_in, out_list) = get_data_test()
test_out = model.predict(test_in)
out_num = 0
for test_num in test_out:
    test_name = out_list[out_num]
    test_dis = test_name[7:11]
    pr_name = test_name[0:4]
    test_name = 'pr' + pr_name + '_dis' + test_dis + '.bmp'
    path_out = r"../data/experiment_data/prediction_out/" + test_name
    cv2.imwrite(path_out, test_num)
    out_num = out_num + 1

print('it is over')
