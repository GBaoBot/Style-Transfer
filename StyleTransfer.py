import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, AveragePooling2D, MaxPooling2D, Conv2D
from tensorflow.keras.models import Model, Sequential

import tensorflow.keras.backend as K
from scipy.optimize import fmin_l_bfgs_b
from ContentModel import VGG16_AvgPool, VGG16_AvgPool_Cutoff, unpreprocess, scale_img
from StyleModel import gram_matrix, style_loss, run

def preprocess_img(img):
    img_ = image.img_to_array(img)
    img_ = np.expand_dims(img_, axis=0)
    img_ = preprocess_input(img_)

    return img_

with tf.device('/device:GPU:0'):
    content_img_path = '/home/notomo/Documents/VSC/Models/Data/elephant.jpg'
    style_img_path = '/home/notomo/Documents/VSC/Models/Data/starrynight.jpg'

    # Input content image
    content_img = image.load_img(content_img_path)
    content_img = preprocess_img(content_img)

    # Collect shape (3 dimensions) and batch_shape(4 dimensions)
    batch_shape = content_img.shape
    shape = content_img.shape[1:]
    h, w = content_img.shape[1:3]

    # Input style image
    style_img = image.load_img(style_img_path, target_size=(h, w))
    plt.imshow(style_img)
    plt.show()
    
    # style_img = preprocess_img(style_img)

    # vgg = VGG16_AvgPool(shape)

    # # Content Model
    # content_model = Model(vgg.input, vgg.layers[13].get_output_at(0))
    # content_target = K.variable(content_model.predict(content_img))

    # # Style Model
    # symbolic_style_output = [layer.get_output_at(1) for layer in vgg.layers if layer.name.endswith('conv1')]
    # multi_style_model = Model(vgg.input, symbolic_style_output)

    # style_target = [K.variable(y) for y in multi_style_model.predict(style_img)]
    # style_weights = [0.2,0.4,0.3,0.5,0.2]

    # loss_content = K.mean(K.square(content_target - content_model.output))
    # loss_style = 0
    # for output, target, w in zip(symbolic_style_output, style_target, style_weights):
    #     loss_style += w * style_loss(target[0], output[0])

    # loss = loss_content + loss_style
    # grads = K.gradients (loss, vgg.input)

    # get_loss_and_grads = K.function(inputs=[vgg.input],
    #                                 outputs=[loss] + grads)

    # def get_loss_and_grads_wrapper(x_vec):
    #     l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
    #     return l.astype(np.float64), g.flatten().astype(np.float64)
    
    # losses = []
    # final_img = run(get_loss_and_grads_wrapper, 10, batch_shape)

    # plt.imshow(scale_img(final_img[0]))
    # plt.show()