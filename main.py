import numpy as np
import tensorflow
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Activation
from keras.utils import np_utils
from keras.layers import Conv2D,Input, add, Average,Concatenate
from keras.models import Model
from keras.optimizers import SGD

'''
model = Sequential()
model.add(Convolution2D(4, kernel_size=(5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))


model.add(Dense(2))
model.add(Activation('softmax'))
'''
def model():
    inlayer =Input(shape=(28, 28, 1))

    hidden_layer1_1 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='same',use_bias=True, input_shape=(28, 28, 1))(inlayer)
    hidden_layer1_2 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='same',use_bias=True, input_shape=(28, 28, 1))(inlayer)
    hidden_layer1_3 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='same',use_bias=True, input_shape=(28, 28, 1))(inlayer)
    hidden_layer1_4 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='same',use_bias=True, input_shape=(28, 28, 1))(inlayer)

    hidden_layer2_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(hidden_layer1_1)
    hidden_layer2_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(hidden_layer1_2)
    hidden_layer2_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(hidden_layer1_3)
    hidden_layer2_4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(hidden_layer1_4)

    avg_1 = Average()([hidden_layer2_1, hidden_layer2_2])
    avg_2 = Average()([hidden_layer2_3, hidden_layer2_4])

    hidden_layer3_1 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='same',use_bias=True)(hidden_layer2_1)
    hidden_layer3_2 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='same',use_bias=True)(avg_1)
    hidden_layer3_3 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='same',use_bias=True)(avg_1)
    hidden_layer3_4 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='same',use_bias=True)(hidden_layer2_2)
    hidden_layer3_5 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='same',use_bias=True)(avg_1)
    hidden_layer3_6 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='same',use_bias=True)(avg_1)
    hidden_layer3_7 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='same',use_bias=True)(hidden_layer2_3)
    hidden_layer3_8 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='same',use_bias=True)(avg_2)
    hidden_layer3_9 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='same',use_bias=True)(avg_2)
    hidden_layer3_10 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='same',use_bias=True)(hidden_layer2_4)
    hidden_layer3_11 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='same',use_bias=True)(avg_2)
    hidden_layer3_12 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='same',use_bias=True)(avg_2)


    hidden_layer4_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(hidden_layer3_1)
    hidden_layer4_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(hidden_layer3_2)
    hidden_layer4_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(hidden_layer3_3)
    hidden_layer4_4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(hidden_layer3_4)
    hidden_layer4_5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(hidden_layer3_5)
    hidden_layer4_6 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(hidden_layer3_6)
    hidden_layer4_7 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(hidden_layer3_7)
    hidden_layer4_8 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(hidden_layer3_8)
    hidden_layer4_9 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(hidden_layer3_9)
    hidden_layer4_10 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(hidden_layer3_10)
    hidden_layer4_11 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(hidden_layer3_11)
    hidden_layer4_12 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(hidden_layer3_12)


    interTensor = Concatenate()([hidden_layer4_1,
                                   hidden_layer4_2 ,
                                   hidden_layer4_3 ,
                                   hidden_layer4_4 ,
                                   hidden_layer4_5 ,
                                   hidden_layer4_6 ,
                                   hidden_layer4_7 ,
                                   hidden_layer4_8 ,
                                   hidden_layer4_9 ,
                                   hidden_layer4_10,
                                   hidden_layer4_11,
                                   hidden_layer4_12])

    interTensor = Dense(4)(interTensor)
    interTensor = Flatten()(interTensor)
    outputTensor = Dense(10, activation='softmax')(interTensor)

    optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

    model = Model(inputs=inlayer, outputs=outputTensor)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

