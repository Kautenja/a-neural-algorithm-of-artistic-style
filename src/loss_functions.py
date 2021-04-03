"""Methods representing various loss functions."""
from tensorflow.keras import backend as K


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


def gram(x):
    """
    Return a gram matrix for the given input matrix.

    Args:
        x: the matrix to calculate the gram matrix of

    Returns:
        the gram matrix of x

    """
    # use the keras function to access shape opposed to the instance member.
    # this allows backward compatibility with TF1.2.1 (the version in conda)
    shape = K.shape(x)
    # flatten the 3D tensor by converting each filter's 2D matrix of points
    # to a vector. thus we have the matrix:
    # [filter_width x filter_height, num_filters]
    F = K.reshape(x, (shape[0] * shape[1], shape[2]))
    # take inner product over all the vectors to produce the Gram matrix over
    # the number of filters
    return K.dot(K.transpose(F), F)


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
    Ml2 = int(style.shape[0] * style.shape[1])**2
    # N_l is the number of distinct filters in the layer
    Nl2 = int(style.shape[2])**2

    # take the squared euclidean distance between the gram matrices of both
    # the style and combination image. divide by the constant scaling factor
    # based on parameterized sizes
    return K.sum(K.square(gram(style) - gram(combination))) / (4 * Nl2 * Ml2)


def total_variation_loss(x, kind='isotropic'):
    """
    Return the total variation loss for the image x.

    Args:
        x: the image tensor to return the variation loss of
        kind: the kind of total variation loss to use (default 'anisotropic')

    Returns:
        the total variation loss of the image x

    """
    # store the dimensions for indexing from the image x
    h, w = x.shape[1], x.shape[2]
    if kind == 'anisotropic':
        # take the absolute value between this image, and the image one pixel
        # down, and one pixel to the right. take the absolute value as
        # specified by anisotropic loss
        a = K.abs(x[:, :h-1, :w-1, :] - x[:, 1:, :w-1, :])
        b = K.abs(x[:, :h-1, :w-1, :] - x[:, :h-1, 1:, :])
        # add up all the differences
        return K.sum(a + b)
    elif kind == 'isotropic':
        # take the absolute value between this image, and the image one pixel
        # down, and one pixel to the right. take the square root as specified
        # by isotropic loss
        a = K.square(x[:, :h-1, :w-1, :] - x[:, 1:, :w-1, :])
        b = K.square(x[:, :h-1, :w-1, :] - x[:, :h-1, 1:, :])
        # take the vector square root of all the pixel differences, then sum
        # them all up
        return K.sum(K.pow(a + b, 2))
    else:
        # kind can only be two values, raise an error on unexpected kind value
        raise ValueError("`kind` should be 'anisotropic' or 'isotropic'")


# explicitly define the outward facing API of this module
__all__ = [
    content_loss.__name__,
    style_loss.__name__,
    total_variation_loss.__name__,
]
