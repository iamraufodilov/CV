# load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, InputLayer, BatchNormalization
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow
from sklearn.metrics import confusion_matrix


class Model():

    def create_model(self, model_name):
        if model_name == 'basic':

            # build sequential model
            model = Sequential()
            model.add(InputLayer(input_shape = (224,224, 3)))

            # 1st conv block
            model.add(Conv2D(25, (5,5), activation = 'relu', strides = (1, 1), padding = 'same'))
            model.add(MaxPool2D(pool_size = (2, 2), padding = 'same'))

            # 2nd block
            model.add(Conv2D(50, (5, 5), activation = 'relu', strides = (2, 2), padding = 'same'))
            model.add(MaxPool2D(pool_size = (2, 2), padding = 'same'))
            model.add(BatchNormalization())

            # 3rd bloack
            model.add(Conv2D(70, (3, 3), activation = 'relu', strides = (2, 2), padding = 'same'))
            model.add(MaxPool2D(pool_size = (2, 2), padding = 'valid'))
            model.add(BatchNormalization())

            # Fully Connected Layer
            model.add(Flatten())
            model.add(Dense(units = 100, activation = 'relu'))
            model.add(Dense(units = 100, activation = 'relu'))
            model.add(Dropout(0.25))

            # Output layer
            model.add(Dense(units = 1, activation = 'sigmoid'))

            model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

            return model

        elif model_name == 'vgg16':
            bae_model = VGG16(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet')
            for layer in base_model.layers:
                layer.trainable = False

            # flatten the output layer to 2 dimension
            x = Flatten()(base_model.output)
            x = Dense(512, activation = 'relu')(x)
            x = Dropout()(x)
            x = Dense(1, activation = 'sigmoid')(x)

            model = tensorflow.keras.models.Model(base_model.input, x)

            model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

            return model
        else:
            print("Please choose model type")

    # train the model
    def train(self, feature, label, model, epochs):
        model.fit(feature, label, epochs = epochs)
        return model

    def evaluation(self, feature, label, model):
        y_predicted = model.predict(feature)
        accuracy = confusion_matrix(label, y_predicted)
        print("Here is confusion matrix result to evaluate result: {}".format(accuracy))


# finished 