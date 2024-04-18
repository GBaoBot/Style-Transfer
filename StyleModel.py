import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, AveragePooling2D, MaxPooling2D, Conv2D
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K

from scipy.optimize import fmin_l_bfgs_b
from ContentModel import VGG16_AvgPool, unpreprocess, scale_img


# Used for calculating style_loss
def gram_matrix(img):
  # input is (H, W, C) (C = # feature maps)
  # we first need to convert it to (C, H*W)
  X = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))
  
  # now, calculate the gram matrix
  # gram = XX^T / N
  # the constant is not important since we'll be weighting these
  G = K.dot(X, K.transpose(X)) / img.get_shape().num_elements()
  return G


# Calculate style_loss
def style_loss(target, actual):
    '''
    input must be in shape of (H, W, C)
    '''
    loss = K.mean(K.square(gram_matrix(target) - gram_matrix(actual)))
    return loss


# function for run optimizing the generated image
def run(fn, epochs, batch_shape):
    t0 = time.time()
    losses = []
    x = np.random.rand(np.prod(batch_shape))
    for i in range(epochs):
        x, l, _ = fmin_l_bfgs_b(func=fn,
                                x0=x,
                                maxfun=20)
        x = np.clip(x, -127, 127)
        print("iter=%s, loss=%s" % (i, l))
        losses.append(l)

    print("duration:", time.time() - t0)
    plt.plot(losses)
    plt.show()

    newimg = x.reshape(*batch_shape)
    final_img = unpreprocess(newimg)
    return final_img


if __name__ == '__main__':
    with tf.device('/device:GPU:0'):
        path = '/home/notomo/Documents/VSC/Models/Data/starrynight.jpg'
        img = image.load_img(path)

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        batch_shape = x.shape
        shape = x.shape[1:]
        print(shape)
        vgg = VGG16_AvgPool(shape)

        # Create model with multi-outputs
        symbolic_conv_outputs = [layer.output for layer in vgg.layers if layer.name.endswith('conv1')]
        multi_output_model = Model(vgg.input, symbolic_conv_outputs)

        # Run style image through model
        style_layer_outputs = [K.variable(y) for y in multi_output_model.predict(x)]

        # Calculate loss between outputs of model and predicted style images
        loss = 0
        for style, actual in zip(style_layer_outputs, symbolic_conv_outputs):
            loss += style_loss(style[0], actual[0])

        # Gradients for optimizing
        grads = K.gradients(loss, multi_output_model.input)

        # Function returns loss and grads, required for fmin_l_bfgs_b
        get_loss_and_grads = K.function(inputs = [multi_output_model.input],
                                    outputs = [loss] + grads)

        def get_loss_and_grads_wrapper(x_vec):
            l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
            return l.astype(np.float64), g.flatten().astype(np.float64)

        final_img = run(get_loss_and_grads_wrapper, 10, batch_shape)
        plt.imshow(scale_img(final_img[0]))
        plt.show()