"""An implementation of a basic gradient descent algorithm."""
from typing import Callable
import numpy as np


class SGD(object):
    """Classic Stochastic Gradient Descent."""

    def __init__(self, learning_rate: float=1e-4) -> None:
        """
        Initialize a new Stochastic Gradient Descent optimizer.

        Args:
            learning_rate: how fast to adjust the parameters (dW)

        Returns: None
        """
        self.learning_rate = learning_rate

    def __repr__(self) -> str:
        """Return an executable string representation of this object."""
        return '{}(learning_rate={})'.format(*[
            self.__class__.__name__,
            self.learning_rate
        ])

    def minimize(self,
                 X: np.ndarray,
                 shape: tuple,
                 loss_grads: Callable,
                 iterations: int=1000,
                 callback: Callable=None):
        """
        Reduce the loss geanerated by X by moving it based on its gradient.

        Args:
            X: the input value to adjust to minimize loss
            shape: the shape to coerce X to
            loss_grads: a callable method that returns loss and gradients
                        given some input
            iterations: the number of iterations of optimization to perform
            callback: an optional callback method to receive image updates

        Returns: an optimized X about the loss and gradients given
        """
        for i in range(iterations):
            # pass the input through the loss function and generate gradients
            loss_i, grads_i = loss_grads([X])
            # move the input based on the gradients and learning rate
            X -= self.learning_rate * grads_i
            # pass the values to the callback if any
            if callable(callback):
                callback(X, i)

        return X


__all__ = ['SGD']
