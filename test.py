from keras.models import Model
import numpy as np
import gzip, struct, random
from keras.models import load_model
import matplotlib.pylab as plt
from keras.utils import to_categorical

# load the data from the files

with gzip.open('t10k-labels-idx1-ubyte.gz') as f:
    magic, num = struct.unpack(">II", f.read(8))
    test_labels = np.fromstring(f.read(), dtype=np.int8)

with gzip.open('t10k-images-idx3-ubyte.gz') as f:
    magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
    test_images = np.fromstring(f.read(), dtype=np.uint8).reshape(num, rows, cols, 1)

# Normalize and encode the labeling of data set
test_labels_encoded = to_categorical(test_labels)
test_images = test_images / 255
n = int(len(test_images) / 2)
X_valid, y_valid = test_images[:n], test_labels_encoded[:n]
X_test, y_test = test_images[n:], test_labels_encoded[n:]

# load the model form the saved model
cnn_model = load_model('my_model.h5')


# Evaluate the model
cnn_scores = cnn_model.evaluate(X_test, y_test, verbose=0)

print("CNN Scores: ", (cnn_scores))
print("CNN Error: %.2f%%" % (100 - cnn_scores[1] * 100))

plt.figure(figsize=(14, 5))
plt.plot(cnn_history.history['categorical_accuracy'][3:], '-o', label='train')
plt.plot(cnn_history.history['val_categorical_accuracy'][3:], '-o', label='test')
plt.legend()
plt.title('CNN Accuracy');

