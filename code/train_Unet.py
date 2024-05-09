# Created Time    : 2024/2/25 16:29
# Author  : Bozhen Zhou (zbz22@mails.tsinghua.edu.cn)
# Tips    : 1. Make sure all the data paths are right. They are in
#              line 34, line 213, line 231 and 232.
#           2. You need to change these data and the data paths to train different models.
#           3. When no validation data involved, change the callback in the last line from "my_callback"
#              to "model_checkpoint".
#           4. Change the model name in "model_name" in different trainings. Change the "batch_size"
#              if your GPU memory is not enough.

import os
import cv2
import numpy as np
from tensorflow.keras.callbacks import Callback
from keras.layers import *
from keras.models import *
from function.vgg import VGG
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.optimizers import *


def custom_sort_key(item):
    parts = item.split('dis')
    a = parts[0][1]
    b = parts[0][3]
    dis = parts[1][0:4]
    return a, b, dis


# make sure the path in the "path_in" below is correct, inside which should be your validation label
def get_data_val_out():
    path_in = r"../data/experiment_data/val_data"
    t_in = []
    img_in_list = os.listdir(path_in)
    img_in_list_new = sorted(img_in_list, key=custom_sort_key)
    for name in img_in_list_new:
        img_in = cv2.imread(path_in + '/' + name, cv2.IMREAD_GRAYSCALE)
        img_in = np.resize(img_in, (256, 256, 1))
        t_in.append(img_in)
    return np.array(t_in), img_in_list_new


validation_data = get_data_val_out()[0]


# convert the reference result into dictionary format
def val_out_to_dict(val_out):
    par_info = np.nonzero(val_out)
    true_diction = dict()
    par_num = len(par_info[0])
    panju = 0
    dict_value = []
    for i in range(par_num):
        dis_index = par_info[0][i]
        if (dis_index // 61) != panju:
            dict_value = []
        panju = dis_index // 61
        a_num = dis_index // (61 * 4) + 1
        a_yushu = dis_index % (61 * 4)
        b_num = a_yushu // 61 + 1
        b_yushu = a_yushu % 61
        dis = b_yushu * 50 + 1800
        Y = par_info[1][i]
        X = par_info[2][i]
        diam = val_out[dis_index, Y, X, 0]
        dict_value.append([Y, X, dis, diam])
        dict_name = "a%db%d" % (a_num, b_num)
        true_diction[dict_name] = dict_value
    return true_diction


# output processing algorithm: convert the prediction images into particle field information in dictionary format
def pre_to_dict(pre_pack):
    pre_dict = dict()
    total_num = pre_pack.shape[0]
    for i in range(total_num):
        a_num = i // (61 * 4) + 1
        a_yushu = i % (61 * 4)
        b_num = a_yushu // 61 + 1
        b_yushu = a_yushu % 61
        dis = b_yushu * 50 + 1800
        dict_name = "a%db%d" % (a_num, b_num)
        if dis == 1800:
            dict_value = []
        this_img = pre_pack[i]
        this_img = np.resize(this_img, (256, 256))
        fi = this_img
        fi[fi < 50] = 0
        fi[fi > 150] = 0
        panju = np.all(fi == 0)
        zuida = np.max(fi)
        final_pre_result = []
        pre_num = 0
        if not panju:
            _, bw_fi = cv2.threshold(fi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(bw_fi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            show_i = i
            num_contour = len(contours)
            result = []
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    grey_value = np.mean(fi[cY - 1:cY + 2, cX - 1:cX + 2])
                    if 75 < grey_value < 125:
                        pre_num += 1
                        dict_value.append([cX, cY, dis, round(grey_value, 1)])
                        result.append([cX, cY, dis, int(grey_value)])
                        pre_dict[dict_name] = dict_value
            if result:
                final_pre_result.extend(result)
    return pre_dict


# custom evaluation metric: get the exaction rate, JI, and the number of prediction particles
def diy_metric(y_true, y_pred):
    diction = val_out_to_dict(y_true)
    y_pred = cv2.convertScaleAbs(y_pred)
    pre_dict = pre_to_dict(y_pred)
    sum_truth_num = 0
    sum_pre_num = 0
    positive_num = 0
    omitted_num = 0
    for aa in range(4):
        aa += 1
        for bb in range(4):
            bb += 1
            dict_name = "a%db%d" % (aa, bb)
            true_value = diction[dict_name]
            true_num = len(true_value)  # particle number in the reference result in this image
            if pre_dict.get(dict_name) is None:
                sum_truth_num += true_num
                omitted_num += true_num
                continue
            pre_value = pre_dict[dict_name]
            pre_num = len(pre_value)  # particle number in the prediction result in this image
            sum_truth_num += true_num  # get the amount of all the reference particles
            sum_pre_num += pre_num  # get the amount of all the prediction particles
            for i in range(pre_num):
                pre_par = pre_value[i]
                X = pre_par[0]
                Y = pre_par[1]
                Z = pre_par[2]
                for j in range(true_num):
                    true_par = true_value[j]
                    X0 = true_par[0]
                    Y0 = true_par[1]
                    Z0 = true_par[2]
                    if (abs(X - X0) < 10) & (abs(Y - Y0) < 10) & (abs(Z - Z0) < 151):
                        positive_num += 1
            for i in range(true_num):
                true_par = true_value[i]
                X0 = true_par[0]
                Y0 = true_par[1]
                Z0 = true_par[2]
                for j in range(pre_num):
                    pre_par = pre_value[j]
                    X = pre_par[0]
                    Y = pre_par[1]
                    Z = pre_par[2]
                    if (abs(X - X0) < 14) & (abs(Y - Y0) < 14) & (abs(Z - Z0) < 151):
                        break
                    if j == pre_num - 1:
                        omitted_num += 1

    negative_num = sum_pre_num - positive_num
    extract_rate = (1 - omitted_num / sum_truth_num)
    extract_rate_per = "{:.2%}".format(extract_rate)
    JI = positive_num / (sum_pre_num + omitted_num)
    JI = round(JI, 3)
    return JI, extract_rate_per, sum_pre_num


# custom callback function to control the training
class MyCallback(Callback):
    def __init__(self, validation_data, model_name):
        self.validation_data = validation_data
        self.best_point = float(-1)
        self.best_model = None
        self.model_name = model_name

    # define a criterion "Max Point" to determine whether it is raised based on the value of the point
    def point_improved(self, point):
        if point > self.best_point:
            return True
        else:
            return False

    def on_epoch_end(self, epoch, logs=None):
        val_data = self.validation_data
        results = diy_metric(val_data[1], self.model.predict(val_data[0]))
        if 60 < results[2] < 999:  # Point is not 0 when the particle number is over 60
            point = results[0] + 0.01 * float(results[1][0:5])
        else:
            point = 0
        print(f'JI: {results[0]}, extract_rate: {results[1]}, sum_pre_num: {results[2]}')
        print(f'point: {point}')

        # when the Point improves, save the model
        if self.point_improved(point):
            self.model.save(self.model_name)
            self.best_point = point
            print(f"point improved, saving to {self.model_name}")

        print()


# get the validation input data, make sure the "path_in" below is correct
def get_data_test():
    path_in = r"../data/experiment_data/test_in"
    t_in = []
    img_in_list = os.listdir(path_in)
    img_in_list_new = sorted(img_in_list, key=custom_sort_key)
    for name in img_in_list_new:
        img_in = cv2.imread(path_in + '/' + name, cv2.IMREAD_GRAYSCALE)
        img_in = np.resize(img_in, (256, 256, 1))
        t_in.append(img_in)
    return np.array(t_in), img_in_list_new


(test_in, out_list) = get_data_test()
val_data_in = test_in  # ndarray:(976,256,256,1)
val_data_out = validation_data


# get the training data, make sure the paths in "path_in" and "path_out" are correct
def get_data():
    path_in = r"../data/experiment_data/train_in"
    path_out = r"../data/experiment_data/train_out"
    t_in = []
    t_out = []
    img_in_list = os.listdir(path_in)
    img_out_list = os.listdir(path_out)
    for name in img_in_list:
        holo_count = name.split('_')[0][2:]  # group number
        holo_distance = name.split('_')[-1][2:6]  # propagation distance
        img_in = cv2.imread(path_in + '/' + name, cv2.IMREAD_GRAYSCALE)
        img_in = np.resize(img_in, (256, 256, 1))
        for name_gt in img_out_list:
            check_name_gt = name_gt.split("_")[0][2:]  # group number
            check_dis = name_gt.split('_')[-1][2:6]  # propagation distance
            if check_name_gt == holo_count and check_dis == holo_distance:
                img_out = cv2.imread(path_out + '/' + name_gt, cv2.IMREAD_GRAYSCALE)
                img_out = np.resize(img_out, (256, 256, 1))
                t_in.append(img_in)
                t_out.append(img_out)
    return np.array(t_in), np.array(t_out)


# the modified U-Net
def Unet(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)
    # -------------------------------#
    #   get five feature layers
    #   feat1   256,256,64
    #   feat2   128,128,128
    #   feat3   64,64,256
    #   feat4   32,32,512
    #   feat5   16,16,512
    # -------------------------------#
    feat1, feat2, feat3, feat4, feat5 = VGG(inputs)

    channels = [64, 128, 256, 512]
    P5_up = BatchNormalization()(feat5, training=True)
    P5_up = UpSampling2D(size=(2, 2))(P5_up)  # 16,16,512 -> 32,32,512
    P4 = Concatenate(axis=3)([feat4, P5_up])  # 32,32,512 + 32,32,512 -> 32,32,1024
    P4 = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        P4)  # 32,32,1024 -> 32,32,512
    P4 = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        P4)  # 32,32,512 -> 32,32,512
    P4_up = BatchNormalization()(P4, training=True)
    P4_up = UpSampling2D(size=(2, 2))(P4_up)  # 32,32,512 -> 64,64,512
    P3 = Concatenate(axis=3)([feat3, P4_up])  # 64,64,256 + 64,64,512 -> 64,64,768
    P3 = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        P3)  # 64,64,768 -> 64,64,256
    P3 = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        P3)  # 64,64,256 -> 64,64,256
    P3_up = BatchNormalization()(P3, training=True)
    P3_up = UpSampling2D(size=(2, 2))(P3_up)  # 64,64,256 -> 128,128,256
    P2 = Concatenate(axis=3)([feat2, P3_up])  # 128,128,256 + 128,128,128 -> 128,128,384
    P2 = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        P2)  # 128,128,384 -> 128,128,128
    P2 = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        P2)  # 128,128,384 -> 128,128,128
    P2_up = BatchNormalization()(P2, training=True)
    P2_up = UpSampling2D(size=(2, 2))(P2_up)  # 128,128,128 -> 256,256,128
    P1 = Concatenate(axis=3)([feat1, P2_up])  # 256,256,64 + 256,256,128 -> 256,256,192
    P1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        P1)  # 256,256,192 -> 256,256,64
    P1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        P1)  # 256,256,64 -> 256,256,64

    P1 = Conv2D(1, kernel_size=3, strides=1, padding='same')(P1)  # 256,256,64 -> 256,256,1
    model = Model(inputs=inputs, outputs=P1)
    model.compile(optimizer=adam_v2.Adam(lr=1e-5), loss='mse')
    return model


[data_in, data_out] = get_data()
print(data_in.shape)
model = Unet()
model_name = 'trained_model.h5'
model_checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True)
my_log_dir = 'tensorboard_log'  # location
my_tensorboard = TensorBoard(log_dir=my_log_dir)
my_callback = MyCallback([val_data_in, val_data_out], model_name)
model.fit(data_in, data_out, batch_size=16, epochs=100, verbose=2, callbacks=[my_callback])  # model_checkpoint
