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
from slacker import Slacker
from keras.callbacks import Callback
from timeit import default_timer as timer
from datetime import datetime

reporting_channel = 'handwrittendigit'


def report_stats(text, channel):
    """Report training stats"""
    r = slack.chat.post_message(channel=channel, text=text,
                                username='Training Report',
                                icon_emoji=':clipboard:')

    if r.successful:
        return True
    else:
        return r.error


class SlackUpdate(Callback):
    """Custom Keras callback that posts to Slack while training a neural network"""

    def __init__(self, channel):
        self.channel = channel

    def on_train_begin(self, logs={}):
        report_stats(text=f'Training started at {datetime.now()}',
                     channel=reporting_channel)

        self.start_time = timer()
        self.train_acc = []
        self.valid_acc = []
        self.train_loss = []
        self.valid_loss = []
        self.n_epochs = 0

    def on_epoch_end(self, batch, logs={}):

        self.train_acc.append(logs.get('categorical_accuracy'))
        self.valid_acc.append(logs.get('val_categorical_accuracy'))
        self.train_loss.append(logs.get('loss'))
        self.valid_loss.append(logs.get('val_loss'))
        self.n_epochs += 1

        message = f'Epoch: {self.n_epochs} Training Loss: {self.train_loss[-1]:.4f} Train_accuracy: ' \
                  f'{self.train_acc[-1]:.4f}  Validation Loss: {self.valid_loss[-1]:.4f} Valid_accuracy: ' \
                  f'{self.valid_acc[-1]:.4f}'

        report_stats(message, channel=self.channel)

    def on_train_end(self, logs={}):

        best_epoch = np.argmin(self.valid_loss)
        valid_loss = self.valid_loss[best_epoch]
        train_loss = self.train_loss[best_epoch]
        train_acc = self.train_acc[best_epoch]
        valid_acc = self.valid_acc[best_epoch]

        message = f'Trained for {self.n_epochs} epochs. Best epoch was {best_epoch + 1}.'
        report_stats(message, channel=self.channel)
        message = f'Best validation loss = {valid_loss:.4f} Training Loss = {train_loss:.2f} Validation accuracy = ' \
                  f'{100*valid_acc:.2f}%'
        report_stats(message, channel=self.channel)


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

slack_token_id = None
filepath = 'config'
with open(filepath) as fp:
    slack_token_id = fp.readline()


slack = Slacker(slack_token_id)
updater = SlackUpdate(channel=reporting_channel)
cnn_model = model()
cnn_model.save('my_model.h5')
print(cnn_model.summary())

cnn_history = cnn_model.fit(x=train_images, y=train_labels_encoded, batch_size=32, epochs=2000, verbose=2,
                            validation_split=0.0, validation_data=(X_valid, y_valid), shuffle=True, callbacks=[updater],
                            class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None,
                            validation_steps=None)

cnn_scores = cnn_model.evaluate(X_test, y_test, verbose=0)

print("CNN Scores: ", (cnn_scores))
print("CNN Error: %.2f%%" % (100 - cnn_scores[1] * 100))

plt.figure(figsize=(14, 5))
plt.plot(cnn_history.history['categorical_accuracy'][3:], '-o', label='train')
plt.plot(cnn_history.history['val_categorical_accuracy'][3:], '-o', label='validation')
plt.legend()
plt.title('CNN Accuracy');
plt.savefig('accuracy.png')
slack.files.upload(file_='accuracy.png', title="Training Curves", channels='handwrittendigit')