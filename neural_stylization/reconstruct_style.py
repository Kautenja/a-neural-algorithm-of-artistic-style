"""A functional decomposition of the style reconstruction algorithm."""
import numpy as np
from PIL import Image
from typing import Callable
from keras import backend as K
from .vgg19 import VGG_19
from .util.img_util import load_image
from .util.img_util import normalize
from .util.img_util import denormalize
from .util.img_util import image_to_matrix
from .util.img_util import matrix_to_image
from .loss_functions import style_loss
from .optimizers.l_bfgs import L_BFGS


def reconstruct_style(style_path: str,
                      image_shape: tuple=None,
                      layer_names: list=[
                          'block1_conv1',
                          'block2_conv1',
                          'block3_conv1',
                          'block4_conv1',
                          'block5_conv1'
                      ],
                      optimize: Callable=L_BFGS(),
                      iterations: int=10,
                      noise_range: tuple=(0, 1),
                      callback: Callable=None) -> Image:
    """
    Reconstruct the given content image at the given VGG19 layer.

    Args:
        style_path: the path of the style to reconstruct
        layer_name: the layer to reconstruct the content from
        optimizer: the optimizer for minimizing the content loss
        iterations: the number of iterations to run the optimizer
        noise_range: the range of values for initializing random noise
        callback: the callback for iterations of gradient descent

    Returns:
        the reconstructed content image based on the VGG19 response at the
        given layer name

    """
    # load the image with the given shape (or the default shape if there is
    # no shape provided)
    style = load_image(style_path, image_shape)
    # convert the binary image to a 3D NumPy matrix of RGB values
    style = image_to_matrix(style)
    # normalize the image's RGB values about the RGB channel means for the
    # ImageNet dataset
    style = normalize(style)

    # load the style image into Keras as a constant, it never changes
    style = K.constant(style, name='Style')
    # create a placeholder for the trained image, this variable trains
    canvas = K.placeholder(style.shape, name='Canvas')
    # combine the style and canvas tensors along the frame axis (0) into a
    # 4D tensor of shape [2, height, width, channels]
    input_tensor = K.concatenate([style, canvas], axis=0)
    # build the model with the 4D input tensor of style and canvas
    model = VGG_19(include_top=False, input_tensor=input_tensor, pooling='avg')

    # initialize the loss to accumulate iteratively over the layers
    loss = K.variable(0.0)

    # iterate over the list of all the layers that we want to include
    for layer_name in layer_names:
        # extract the layer's out that we have interest in for reconstruction
        layer = model[layer_name]
        # calculate the loss between the output of the layer on the style (0)
        # and the canvas (1). The style loss needs to know the size of the
        # image as well by width (shape[2]) and height (shape[1])
        loss = loss + style_loss(layer[0], layer[1])

    # Gatys et al. use a w_l of 1/5 for their example with the 5 layers. As
    # such, we'll simply and say for any length of layers, just take the
    # average. (mirroring what they did)
    loss /= len(layer_names)

    # calculate the gradients
    grads = K.gradients(loss, canvas)[0]
    # generate the iteration function for gradient descent optimization
    step = K.function([canvas], [loss, grads])

    # generate random noise
    noise = np.random.uniform(*noise_range, size=canvas.shape)

    # optimize the white noise to reconstruct the content
    image = optimize(noise, canvas.shape, step, iterations, callback)

    # clear the Keras session
    K.clear_session()

    # denormalize the image (from ImageNet means) and convert back to binary
    return matrix_to_image(denormalize(image.reshape(canvas.shape)[0]))


__all__ = ['reconstruct_style']
