from base_model import BaseModel
from base_model_loader import BaseModelLoader
from tensorflow.keras.models import model_from_json
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os


class DigitClassifierBuilder(BaseModel):

    def __init__(self, input_shape, num_classes):
        super(DigitClassifierBuilder, self).__init__()
        self.build_model(input_shape, num_classes)
        self.hist = []

    def build_model(self, input_shape, num_classes):
        self.model = keras.models.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(),
                           metrics=['accuracy'])

    def fit(self, x_train, y_train, batch_size, epochs, plot=False):
        self.hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        if plot:

            plt.plot(self.hist.history['accuracy'])
            plt.plot(self.hist.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            # summarize history for loss
            plt.plot(self.hist.history['loss'])
            plt.plot(self.hist.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

    def evaluate(self, x_test, y_test):
        scores = self.model.evaluate(x_test, y_test)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    def save(self, checkpoint_path):
        file_path = checkpoint_path + 'model.json'
        model_json = self.model.to_json()
        with open(file_path, 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights(checkpoint_path+'model.h5')

    def summary(self):
        self.model.summary()


class ModelLoader(BaseModelLoader):
    DIGITS = ['0', '1',
              '2', '3',
              '4', '5',
              '6', '7',
              '8', '9',
              '10']

    def __init__(self, json_model, model_weights):
        super(ModelLoader, self).__init__(json_model, model_weights)
        print("MNIST model loaded")

    def predict(self, image):
        self.pred = self.loaded_json_model(image)
        return ModelLoader.DIGITS[np.argmax(self.pred)]
