import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

class Preprocessing():

    def load_train_data(self, path):
        img_array = []
        img_name_array = []
        for folder in os.listdir(path):
            for file in os.listdir(os.path.join(path, folder)):
                img_path = os.path.join(path, folder, file)
                img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
                img = np.array(img)
                img = img.astype('float32')
                img /= 255
                img_array.append(img)
                img_name_array.append(folder)
        return np.array(img_array), img_name_array
    '''
    def load_test_data(self, path):
        img_array = []
        img_name_array = []
        for folder in os.listdir(path):
            for file in os.listdir(os.path.join(path, folder)):
                img_path = os.path.join(path, folder, file)
                img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
                img = np.array(img)
                img = img.astype('float32')
                img /= 255
                img_array.append(img)
                img_name_array.append(folder)
        return img_array, img_name_array
    '''

    def convert_name(self, class_list):
        class_dict = {k: v for v, k in enumerate(np.unique(class_list))}
        class_name_list = [class_dict[class_list[i]] for i in range(len(class_list))]
        return np.array(class_name_list)

    def random_input(self, class_feature, class_name):
        rnd = np.random.randint(0, 30)
        random_feature = class_feature[rnd]
        random_label = class_name[rnd]
        return random_feature, random_label

# our data has to be loaded