import numpy as np
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.utils import to_categorical
from keras.layers import Input, add, Average, Concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy
import gzip, struct
import matplotlib.pylab as plt

'''
model = Sequential()
model.add(Convolution2D(4, kernel_size=(5, 5), activation='relu', padding='valid', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))


model.add(Dense(2))
model.add(Activation('softmax'))
'''


def model():
    inlayer = Input(shape=(28, 28, 1))

    hidden_layer1_1 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='valid', use_bias=True,
                                    input_shape=(28, 28, 1))(inlayer)
    hidden_layer1_2 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='valid', use_bias=True,
                                    input_shape=(28, 28, 1))(inlayer)
    hidden_layer1_3 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='valid', use_bias=True,
                                    input_shape=(28, 28, 1))(inlayer)
    hidden_layer1_4 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='valid', use_bias=True,
                                    input_shape=(28, 28, 1))(inlayer)

    hidden_layer2_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=None)(hidden_layer1_1)
    hidden_layer2_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=None)(hidden_layer1_2)
    hidden_layer2_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=None)(hidden_layer1_3)
    hidden_layer2_4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=None)(hidden_layer1_4)

    avg_1 = Average()([hidden_layer2_1, hidden_layer2_2])
    avg_2 = Average()([hidden_layer2_3, hidden_layer2_4])

    hidden_layer3_1 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='valid', use_bias=True)(
        hidden_layer2_1)
    hidden_layer3_2 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='valid', use_bias=True)(avg_1)
    hidden_layer3_3 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='valid', use_bias=True)(avg_1)
    hidden_layer3_4 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='valid', use_bias=True)(
        hidden_layer2_2)
    hidden_layer3_5 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='valid', use_bias=True)(avg_1)
    hidden_layer3_6 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='valid', use_bias=True)(avg_1)
    hidden_layer3_7 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='valid', use_bias=True)(
        hidden_layer2_3)
    hidden_layer3_8 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='valid', use_bias=True)(avg_2)
    hidden_layer3_9 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='valid', use_bias=True)(avg_2)
    hidden_layer3_10 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='valid', use_bias=True)(
        hidden_layer2_4)
    hidden_layer3_11 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='valid', use_bias=True)(avg_2)
    hidden_layer3_12 = Convolution2D(1, kernel_size=(5, 5), activation='relu', padding='valid', use_bias=True)(avg_2)

    hidden_layer4_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=None)(hidden_layer3_1)
    hidden_layer4_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=None)(hidden_layer3_2)
    hidden_layer4_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=None)(hidden_layer3_3)
    hidden_layer4_4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=None)(hidden_layer3_4)
    hidden_layer4_5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=None)(hidden_layer3_5)
    hidden_layer4_6 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=None)(hidden_layer3_6)
    hidden_layer4_7 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=None)(hidden_layer3_7)
    hidden_layer4_8 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=None)(hidden_layer3_8)
    hidden_layer4_9 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=None)(hidden_layer3_9)
    hidden_layer4_10 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=None)(
        hidden_layer3_10)
    hidden_layer4_11 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=None)(
        hidden_layer3_11)
    hidden_layer4_12 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=None)(
        hidden_layer3_12)

    interTensor = Concatenate()([hidden_layer4_1,
                                 hidden_layer4_2,
                                 hidden_layer4_3,
                                 hidden_layer4_4,
                                 hidden_layer4_5,
                                 hidden_layer4_6,
                                 hidden_layer4_7,
                                 hidden_layer4_8,
                                 hidden_layer4_9,
                                 hidden_layer4_10,
                                 hidden_layer4_11,
                                 hidden_layer4_12])
    interTensor = Dropout(0.25)(interTensor)
    interTensor = Dense(1)(interTensor)
    interTensor = Flatten()(interTensor)
    interTensor = Dropout(0.25)(interTensor)
    outputTensor = Dense(10, activation='softmax')(interTensor)

    optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

    model = Model(inputs=inlayer, outputs=outputTensor)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=[categorical_accuracy])
    return model


with gzip.open('train-labels-idx1-ubyte.gz') as f:
    magic, num = struct.unpack(">II", f.read(8))
    train_labels = np.fromstring(f.read(), dtype=np.int8)

with gzip.open('train-images-idx3-ubyte.gz') as f:
    magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
    train_images = np.fromstring(f.read(), dtype=np.uint8).reshape(num, rows, cols, 1)

with gzip.open('t10k-labels-idx1-ubyte.gz') as f:
    magic, num = struct.unpack(">II", f.read(8))
    test_labels = np.fromstring(f.read(), dtype=np.int8)

with gzip.open('t10k-images-idx3-ubyte.gz') as f:
    magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
    test_images = np.fromstring(f.read(), dtype=np.uint8).reshape(num, rows, cols, 1)

train_labels_encoded = to_categorical(train_labels)
test_labels_encoded = to_categorical(test_labels)

train_images = train_images / 255
test_images = test_images / 255


n = int(len(test_images) / 2)
X_valid, y_valid = test_images[:n], test_labels_encoded[:n]
X_test, y_test = test_images[n:], test_labels_encoded[n:]


cnn_model = model()
print(cnn_model.summary())

cnn_history = cnn_model.fit(x=train_images, y=train_labels_encoded, batch_size=32, epochs=2000, verbose=2, callbacks=None,
                            validation_split=0.0, validation_data=(X_valid, y_valid), shuffle=True, class_weight=None,
                            sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)

cnn_scores = cnn_model.evaluate(X_test, y_test, verbose=0)

print("CNN Scores: ", (cnn_scores))
print("CNN Error: %.2f%%" % (100 - cnn_scores[1] * 100))

plt.figure(figsize=(14, 5))
plt.plot(cnn_history.history['categorical_accuracy'][3:], '-o', label='train')
plt.plot(cnn_history.history['val_categorical_accuracy'][3:], '-o', label='test')
plt.legend()
plt.title('CNN Accuracy');
