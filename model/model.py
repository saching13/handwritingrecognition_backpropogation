import numpy as np

EPOCHS = 1000000


class Model:
    def __init__(self, input_shape, lr=0.01, momentum=None, l2_lambda=0.1):
        self.input_shape = input_shape
        self.lr = lr
        self.momentum = momentum
        self.l2_lambda = l2_lambda
        self.layers = []
        self.trained_samples = 0

    def add(self, layer):
        last_shape = self.input_shape if len(self.layers) == 0 else self.layers[-1].get_output_shape()
        layer.set_input_shape(last_shape)
        self.layers.append(layer)


class layers:
    def __init__(self, input_shape, weights, type , Sublayers):
        self.input_shape = input_shape
        self.weights = lr
        self.momentum = momentum
        self.l2_lambda = l2_lambda
        self.layers = []


def build_model():
    model = seq(input_shape=(28, 28, 1), lr=LR, momentum=MOMENTUM, l2_lambda=L2_LAMBDA)

    model.add(conv2d.conv2d(filters=4))
    model.add(relu.relu())
    # model.add(conv2d.conv2d(filters=4))
    # model.add(relu.relu())
    # model.add(maxpooling2d.maxpooling2d(pool_size=2, stride=2))
    model.add(flatten.flatten())
    # model.add(dropout.dropout(0.5))
    model.add(fc.fc(units=10))
    model.add(softmax.softmax())

    return model
