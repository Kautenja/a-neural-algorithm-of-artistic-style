"""Methods representing various loss functions."""
import numpy as np
from keras import backend


def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram


def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))


def style_loss(style, combination, width, height):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = height * width
    return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(canvas):
    h = canvas.height
    w = canvas.width
    a = backend.square(canvas.output[:, :h-1, :w-1, :] - canvas.output[:, 1:, :w-1, :])
    b = backend.square(canvas.output[:, :h-1, :w-1, :] - canvas.output[:, :h-1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))


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
