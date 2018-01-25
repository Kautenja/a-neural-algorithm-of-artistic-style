"""Methods representing various loss functions."""
import numpy as np
from keras import backend as K
from tensorflow import tensordot


def gram(x):
    """
    Return a gram matrix for the given input matrix.

    Args:
        x: the matrix to calculate the gram matrix of

    Returns: the gram matrix of x
    """
    # the gram matrix is defined as the inner product of a tensor over the
    # i,j pairing.
    return tensordot(x, K.transpose(x), 2)


def content_loss(content, combination):
    """
    Return the content loss between the content and combinations tensors.

    Args:
        content: the original content tensor to measure loss from
        combination: the combination image to reduce the loss of

    Returns: the scalar loss between `content` and `combination`
    """
    # squared euclidean distance, exactly how it is in the paper
    return 0.5 * K.sum(K.square(combination - content))


def style_loss(style, combination, width, height, channels=3):
    """
    Return the style loss for the given style and combination matrices.

    Args:
        style: the original style image to measure loss from
        combination: the combination image to reduce the loss of
        width: the width of the image
        height: the height of the image
        channels: the number of channels in the image (Default 3, RGB/BGR)

    Retursn: the scalar loss between `style` and `combination`
    """
    # calculate the factor that multiplies by the sum. It's originally a
    # fractional piece of one, but we'll just divide to save the unnecessary
    # extra steps
    factor = 4.0 * channels**2 * (width * height)**2
    # take the squared euclidean distance between the gram matrices of both
    # the style and combination image. Divide this by the factor described
    # above
    return K.sum(K.square(gram(style) - gram(combination))) / factor


def total_variation_loss(canvas):
    h = canvas.height
    w = canvas.width
    a = K.square(canvas.output[:, :h-1, :w-1, :] - canvas.output[:, 1:, :w-1, :])
    b = K.square(canvas.output[:, :h-1, :w-1, :] - canvas.output[:, :h-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


class Evaluator(object):

    def __init__(self, eval_loss_and_grads):
        self.eval_loss_and_grads = eval_loss_and_grads
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


# explicitly export the public API
__all__ = [
    'gram_matrix',
    'content_loss',
    'style_loss',
    'total_variation_loss',
    'Evaluator'
]
