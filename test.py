from keras.models import Model
import numpy as np
import gzip, struct, random
from keras.utils import to_categorical
from keras.layers import Conv2D,Input, add, average, concatenate,Average,Dense,Dropout, Flatten
import pandas as pd

'''array = np.arange(16)
input_a = np.reshape(array, (1, 4, 4))
input_b = np.reshape(array, (1, 4, 4))
print(input_a)
a = Input(shape=(4, 4))
b = Input(shape=(4, 4))

concat = concatenate([a,b,],axis=-1)
concat2 = concatenate([a,b,], axis=1)
hidden_layer2_1 = Dense(4, activation='softmax')(concat2)



model_concat = Model(input=[a, b], output=concat)
model_dot = Model(input=[a, b], output=concat2)
model_cos = Model(input=[a, b], output=hidden_layer2_1)
#model_cos1 = Model(input=a , output=hidden_layer2_4)

print(model_concat.predict([input_a, input_b]))
print(model_dot.predict([input_a, input_b]))
print(model_cos.predict([input_a, input_b]))
#print(model_cos1.predict(input_a))'''

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

encoded = to_categorical(test_labels)
print(encoded.shape)

test_images = test_images / 255

print(test_images[0])