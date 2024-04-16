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

# Disable Eager Execution for Tensorflow 2.x
if tf.__version__.startswith('2'):
  tf.compat.v1.disable_eager_execution()


# Create VGG16 Model with AvgPooling2D 
def VGG16_AvgPool(shape):
    '''
    shape: Input shape
    return: Model VGG16
    '''

    model = VGG16(input_shape=shape, weights='imagenet', include_top=False)

    # Create a new model
    i = model.input
    x = i

    for layer in model.layers:
        if layer.__class__ == MaxPooling2D:
            x = AveragePooling2D()(x)
        else:
            x = layer(x)

    return Model(i, x)


# Create VGG16 Model with specific number of Convolutional Layers
def VGG16_AvgPool_Cutoff(shape, n_conv):
    '''
    shape: input shape
    n_conv: # convolutional layers
    return: Model VGG16
    '''

    if n_conv < 1 or n_conv > 13:
        print('n_conv must be from 1 to 13!')
        return

    model = VGG16_AvgPool(shape)

    output = None
    n = 0

    for layer in model.layers:
        if layer.__class__ == Conv2D:
            n += 1
        if n >= n_conv:
            output = layer.output
            break

    return Model(model.input, output)


# To Show image, we unpreprocess the image
def unpreprocess(img):
    '''
    img: image after preprocessing (VGG16)
    return: image after unpreprocessing
    '''
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]
    return img

def scale_img(x):
    '''
    x: unscaled image
    return: image can be shown with matplotlib
    '''
    x = x - x.min()
    x = x / x.max()
    return x

# For testing model
if __name__ == '__main__':

    # Input Image
    path = '/home/notomo/Documents/VSC/Models/Data/elephant.jpg'
    # path = '/content/elephant.jpg'
    img_content = image.load_img(path)

    # Preprocess Image
    img_content = image.img_to_array(img_content)
    img_content = np.expand_dims(img_content, axis=0) # Image must be contained in batch - 4 dimensions
    img_content = preprocess_input(img_content)

    # Shape
    batch_shape = img_content.shape
    shape = img_content.shape[1:]

    # Generate model with n_conv=11
    content_model = VGG16_AvgPool_Cutoff(shape, n_conv=11)

    # Here, we mostly use Keras backend to get those variables being in Keras system
    # Also, Using Keras backend helps the vars working in tensor format
    target = K.variable(content_model.predict(img_content))

    # MSE between the content image and generated image
    loss = K.mean(K.square(target - content_model.output))

    # Gradients to optimize the generated image (minimize loss)
    grads = K.gradients(loss, content_model.input)

    # Function required for fmin_l_bfgs_b
    # This function input x0 and return loss and gradients
    get_loss_and_grads = K.function(
        inputs=[content_model.input],
        outputs=[loss] + grads)
    
    # Because fmin_l_bfgs_b only accept input x0 in shape of array (1 dimension)
    # But the model accepts batch of image (4 dimensions)
    # This function convert shape of input appropriately, and astype to float64
    def get_loss_and_grad_wrapper(x_vec):
        loss, grad = get_loss_and_grads([x_vec.reshape(*batch_shape)])
        return loss.astype(np.float64), grad.flatten().astype(np.float64)
    
    # Initialize random image
    x = np.random.randn(np.prod(batch_shape))
    losses = []
    # Loop with epochs = 10
    start_time = time.time()
    for i in range(10):
        x, l, _ = fmin_l_bfgs_b(func=get_loss_and_grad_wrapper,
                                x0=x,
                                maxfun=20)
        x = np.clip(x, -127, 127)

        print("iter=%s, loss=%s" % (i, l))
        losses.append(l)

    print('Duration: ', time.time() - start_time)
    plt.plot(losses)
    plt.show()

    # Show generated image
    newimg = x.reshape(*batch_shape)
    newimg = unpreprocess(newimg)
    plt.imshow(scale_img(newimg[0]))
    plt.show()