# Style-Transfer
In this project, I try to implement a neural style transfer model, utilizing model VGG16 and Keras API. The general idea is that we transform a random_image close to both the content_image and the style_image simultaneously, by minimizing the losses between the random_image with content_image and style_image. By this way, we combine all important features of both images at the same time.

In total, there are 3 files: ContentModel.py, StyleModel.py, and StyleTransfer.py. The first 2 files is used to build content model and style model, and test them. All methods in these 2 files would be imported to the StyleTransfer.py to build the complete model. All images will be passed through the model and calculated the losses:

    Content_Loss = MSE(outputOfContentModel(content_image) - outputContentModel(random_image))
    Style_Loss = MSE(outputStyleModel(style_image) - outputStyleModel(random_image))

    Loss = Content_Loss + Style_Loss

## ContentModel.py
It contains methods to build a CNN model that extracts the content_features of the content_image (objects, background,...). Because we only need the features without knowing what it is, so despite implementing VGG, we only use the output of the last Convolutional layers (or nearly last). Although all the methods (functions) in this file would be imported in the main file (StyleTransfer.py) to build a complete model, you can run this ContentModel.py itself to see what its output looks like. (Remember to pass the path of your content_image)

## StyleModel.py
It contains methods to build a CNN model, extracting style_features from style_image. Unlike content model only using the output of the last Conv layer, this model extracts outputs from all conv layers, this allows us to capture all features from low-level to high-level of image's style. After that, we pass them through a Gram Matrix to get output. The Gram Matrix is used to represent the correlation between 2 images and gives us an appropriate loss, by using this with the random_image and the style_image from low-level to high-level convolutional layers (from edge, colors to patterns). Similarly, you can run this file itself to see its outputs.

## StyleTransfer.py
This file imports methods from above 2 files, and uses them to build a complete model. We aim to minimize the total loss during training. 

