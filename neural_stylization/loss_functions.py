"""Methods representing various loss functions."""
import numpy as np
from keras import backend as K


def gram(matrix):
    """
    Return a gram matrix for the given input matrix.

    Args:
        matrix: the matrix to calculate the gram matrix of

    Returns: the gram matrix of `matrix`
    """
    # flatten the 3D tensor by converting each filter's 2D matrix of points
    # to a vector. thus we have the matrix:
    # [filter_width x filter_height, num_filters]
    g = K.reshape(matrix, (matrix.shape[0] * matrix.shape[1], matrix.shape[2]))
    # take inner product over all the vectors to produce the Gram matrix over
    # the number of filters
    g = K.dot(K.transpose(g), g)
    # TODO: test this with an image that is taller than wider to ensure the
    # directionality of the dot operation translates
    return g


def content_loss(content, combination):
    """
    Return the content loss between the content and combinations tensors.

    Args:
        content: the output of a layer for the content image
        combination: the output of a layer for the combination image

    Returns:
        the loss between `content` and `combination`
    """
    # squared euclidean distance, exactly how it is in the paper
    return 0.5 * K.sum(K.square(combination - content))


def style_loss(style, combination):
    """
    Return the style loss between the style and combinations tensors.

    Args:
        style: the output of a layer for the style image
        combination: the output of a layer for the combination image

<<<<<<< HEAD
    Returns: the scalar loss between `style` and `combination`
=======
    Returns:
        the loss between `style` and `combination`
>>>>>>> 079fe4d2e66f75c64f6a51ba537692dd90d8654b
    """
    # M_l is the width times the height of the current layer
    Ml = int(style.shape[0] * style.shape[1])
    # N_l is the number of distinct filters in the layer
    Nl = int(style.shape[2])

    # take the squared euclidean distance between the gram matrices of both
    # the style and combination image. multiply by the coefficient
    return K.sum(K.square(gram(style) - gram(combination))) / (4.0 * Nl**2 * Ml**2)


def total_variation_loss(combination):
    """
    Return the total variation loss for the combination image.

    Args:
        combination: the combination tensor to return the variation loss of

    Returns:
        the total variation loss of the combination tensor
    """
    # store the dimensions for indexing from thee combination
    h, w = combination.shape[1], combination.shape[2]
    # use a 1 pixel tall buffer on the rowwise axis to smooth vertical pixels
    a = K.square(combination[:, :h-1, :w-1, :] - combination[:, 1:, :w-1, :])
    # use a 1 pixel wide buffer on the columnwise axis to smooth horizontal
    # pixels
    b = K.square(combination[:, :h-1, :w-1, :] - combination[:, :h-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# explicitly export the public API
__all__ = [
    'content_loss',
    'style_loss',
    'total_variation_loss'
]
