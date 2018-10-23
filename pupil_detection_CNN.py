import numpy as np
import os
import cv2
import imutils
import random as rnd

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint, History

paths_data=os.listdir("eye_position")

data = []
target = []

for path in paths_data:
    if path[-4:] == ".pgm":
        img = cv2.imread("eye_position/" + path)
        mask = cv2.imread("eye_mask/" + path.replace(".pgm",".png"))

        k1 ,k2 = rnd.randint(-3,3), rnd.randint(-3,3)

        img1 = imutils.rotate(img,k1*5)
        img2 = imutils.rotate(img,k2*5)

        mask1 = imutils.rotate(mask, k1 * 5)
        mask2 = imutils.rotate(mask, k2 * 5)

        data.append(img)
        data.append(img1)
        data.append(img2)

        target.append(mask)
        target.append(mask1)
        target.append(mask2)

data_aug=np.array(data)
target_aug=np.array(target)

data_train = data_aug[:data_aug.shape[0]//2]
data_test = data_aug[data_aug.shape[0]//2:]

target_train = target_aug[:target_aug.shape[0]//2]
target_test = target_aug[target_aug.shape[0]//2:]