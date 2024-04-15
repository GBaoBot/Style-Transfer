import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, AveragePooling2D, MaxPooling2D, Conv2D
from tensorflow.keras.models import Model, Sequential

import tensorflow.keras.backend as K
from scipy.optimize import fmin_l_bfgs_b

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def VGG16_AvgPool(shape):
    model = VGG16(input_shape=shape, weights='imagenet', include_top=False)
    # Create a new model
    i = model.input
    x = i
    for layer in model.layers:
        if isinstance(layer, MaxPooling2D):
            x = AveragePooling2D()(x)
        else:
            x = layer(x)
    return Model(i, x)

def VGG16_AvgPool_Cutoff(shape, n_conv):
    if n_conv < 1 or n_conv > 13:
        print('n_conv must be from 1 to 13!')
        return
    model = VGG16_AvgPool(shape)
    output = None
    n = 0
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            n += 1
        if n >= n_conv:
            output = model.output
            break
    return Model(model.input, output)

def unpreprocess(img):
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]
    return img

def scale_img(x):
    x = x - x.min()
    x = x / x.max()
    return x

path = '/home/notomo/Documents/VSC/Models/Data/elephant.jpg'
img = image.load_img(path)

plt.imshow(img)
plt.show()