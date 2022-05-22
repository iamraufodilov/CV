from train import model
import numpy as np
from preprocessing import Preprocessing
import cv2

def load_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    img = img.astype('float32')
    img /= 255

    return img

img_path = "G:/rauf/STEPBYSTEP/Data2/Custom/horse vs zebra/horse.jpg"
img = load_img(img_path)
prediction = model.predict(img)
print(prediction)