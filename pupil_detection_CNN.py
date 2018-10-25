import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import imutils
import random as rnd
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Input, UpSampling2D, Lambda, Activation, Flatten, Dense
import keras
from keras.losses import mean_squared_error
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K
import tensorflow as tf
from keras.engine.topology import Layer

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self,logs={}):
        self.losses=[]
        self.val_losses=[]

    def on_epoch_end(self,batch,logs={}):
        self.losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])

def custom_loss(y_true,y_pred):
    return mean_squared_error(y_true, y_pred)
    #return mean_squared_error(y_true,y_pred) + (510 - K.sum(K.flatten(y_pred)))

class ThresholdLayer(Layer):
    def __init__(self,**kwargs):
        super(ThresholdLayer,self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name = "kernel",
                                      shape=1,
                                      initializer="uniform",
                                      trainable=True)

        super(ThresholdLayer, self).build(input_shape)

    def call(self, x):
        print(self.kernel)
        return K.tf.to_int32(x<self.kernel)

    def compute_output_shape(self, input_shape):
        return input_shape


paths_data=os.listdir("eye_position")

data = []
target = []

for path in paths_data:
    if path[-4:] == ".pgm":
        img = cv2.imread("eye_position/" + path)
        mask = cv2.imread("eye_mask/" + path.replace(".pgm",".png"))[:,:,0]
        mask = cv2.resize(mask,(int(mask.shape[0]/3),int(mask.shape[1]/3)))

        print(mask.shape)
        k1 ,k2 = rnd.randint(-3,3), rnd.randint(-3,3)

        img1 = imutils.rotate(img,k1*5)
        img2 = imutils.rotate(img,k2*5)
        #print(img.shape)
        mask1 = imutils.rotate(mask, k1 * 5)
        mask2 = imutils.rotate(mask, k2 * 5)

        data.append(img)
        data.append(img1)
        data.append(img2)

        mask = mask.reshape(mask.shape + (1,))
        mask1 = mask1.reshape(mask1.shape + (1,))
        mask2 = mask2.reshape(mask2.shape + (1,))

        target.append(mask.flatten())
        target.append(mask1.flatten())
        target.append(mask2.flatten())

data_aug=np.array(data)
target_aug=np.array(target)
# print(data_aug.shape)
# data_aug = data_aug.reshape(data_aug.shape + (1,))
# target_aug = target_aug.reshape(target_aug.shape + (1,))

data_train = data_aug[:data_aug.shape[0]//2]
data_test = data_aug[data_aug.shape[0]//2:]

target_train = target_aug[:target_aug.shape[0]//2]
target_test = target_aug[target_aug.shape[0]//2:]
print(target_test.shape)
inp = Input(shape=(286,384,3))

x = Conv2D(32,(50,50),padding="same")(inp)
x = MaxPooling2D((2,2))(x)

x = Conv2D(8,(10,10),padding="same")(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(4,(5,5),padding="same")(x)
x = MaxPooling2D((5,5))(x)
# x = Lambda(lambda l: l[:,:286])(x)

x = Conv2D(4,(3,3),padding="same")(x)
x = Flatten()(x)
x = Dense(128 * 95,activation = "softmax", bias = False)(x)



model = Model(inputs = inp, outputs = x)
model.compile(optimizer="Adam", loss="binary_crossentropy", metrics = ["accuracy"])
print(model.summary())

history=LossHistory()
check = ModelCheckpoint("pupil_detection_2.model",monitor="val_loss", verbose=1,save_best_only=True, mode="min")
early_stop = EarlyStopping("val_loss",10,10,1,"min")
#history = model.fit(data_train[0],target_train[0], batch_size=128,epochs=1,callbacks=[history,check], validation_data=(data_test[0], target_test[0]))
model.fit(data_train,target_train, batch_size=128,epochs=50,callbacks=[history,check, early_stop], validation_data=(data_test, target_test))
# model.load_weights("pupil_detection.model")
#
# pred=model.predict(data_train[0].reshape((1,) + data_train[0].shape))[0]
# pred = pred - np.min(pred)
# print(np.min(pred))
# print(np.max(pred))
#
# pred_postproc = np.ndarray(pred.shape)
# print(pred_postproc.shape)
#
# # for i in range(pred_postproc.shape[0]):
# #     for j in range(pred_postproc.shape[1]):
# cv2.imshow("df",pred)
# cv2.imshow("dfs",data_train[0])
# cv2.waitKey(0)

import matplotlib.pyplot as plt

plt.plot(range(len(history.losses)),history.losses,color="blue")
plt.plot(range(len(history.val_losses)),history.val_losses,color="orange")
plt.show()

