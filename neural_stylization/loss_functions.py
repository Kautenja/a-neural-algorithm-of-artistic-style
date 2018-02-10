"""Methods representing various loss functions."""
from keras import backend as K


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


def style_loss(style, combination):
    """
    Return the style loss between the style and combinations tensors.

    Args:
        style: the output of a layer for the style image
        combination: the output of a layer for the combination image

    Returns:
        the loss between `style` and `combination`

    """
    # M_l is the width times the height of the current layer
    Ml_2 = int(style.shape[0] * style.shape[1])**2
    # N_l is the number of distinct filters in the layer
    Nl_2 = int(style.shape[2])**2

    # take the squared euclidean distance between the gram matrices of both
    # the style and combination image. divide by the constant scaling factor
    # based on parameterized sizes
    return K.sum(K.square(gram(style) - gram(combination))) / (4 * Nl_2 * Ml_2)


__all__ = ['content_loss', 'style_loss']
