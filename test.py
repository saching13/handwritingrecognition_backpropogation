from keras.models import Model
import numpy as np
from keras.layers import Conv2D,Input, add, average, concatenate,Average,Dense,Dropout, Flatten

array = np.arange(16)
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
#print(model_cos1.predict(input_a))