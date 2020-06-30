import warnings
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import model_from_json

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

from minst_model import DigitClassifierBuilder

model_path = os.getcwd() + '\models\\'
num_classes = 10
batch_size = 128
epochs = 5
input_shape = (28, 28, 1)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = DigitClassifierBuilder(input_shape, num_classes)

model.fit(x_train, y_train, batch_size, epochs)

model.evaluate(x_test, y_test)

model.save(model_path)

