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

class ThresholdLayer(Layer):
    def __init__(self,**kwargs):
        super(ThresholdLayer,self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name = "kernel",
                                      shape=(1,),
                                      initializer="uniform",
                                      trainable=True)

        super(ThresholdLayer, self).build(input_shape)

    def call(self, x):
        #print(self.kernel[0].eval(session=K.get_session()))
        return

    def compute_output_shape(self, input_shape):
        return (None,1)
        #return input_shape

inp = Input((3,3))
x = ThresholdLayer()(inp)
model = Model(inp,x)
model.compile(optimizer="Adam", loss="mse")

data=np.array([[[1,2,3],[4,5,6],[7,8,9]]])
#target = np.array([[[0,0,0],[1,1,1],[1,1,1]]])
target = np.array([2])
model.fit(data,target,batch_size=1,epochs=10)
