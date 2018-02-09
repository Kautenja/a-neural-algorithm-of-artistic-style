"""An interface to the L-BFGS Algorithm."""
import numpy as np
from typing import Callable
from scipy.optimize import fmin_l_bfgs_b


class L_BFGS(object):
    """L-BFGS Optimization Algorithm."""

    def __init__(self, max_evaluations: int=20) -> None:
        """
        Initialize a new L-BFGS Hill climbing optimizer.

        Args:
            max_evaluations: how fast to adjust the parameters (dW)

        Returns: None
        """
        self.max_evaluations = max_evaluations
        self._loss = None
        self._gradients = None

    def __repr__(self) -> str:
        """Return an executable string representation of this object."""
        return '{}(max_evaluations={})'.format(*[
            self.__class__.__name__,
            self.max_evaluations
        ])

    def loss(self, X):
        """Calculate the loss given some X."""
        # make sure this is called _only after gradients_
        assert self._loss is None
        # calculate and store both the loss and the gradients
        loss, gradients = self.loss_and_gradients(X)
        self._loss = loss
        self._gradients = gradients
        return self._loss

    def gradients(self, _):
        """Calculate the gradients (lazily) given some X."""
        # make sure this is called _only after loss_
        assert self._loss is not None
        # copy the gradients and nullify the cache
        gradients = np.copy(self._gradients)
        self._loss = None
        self._gradients = None
        return gradients

    def minimize(self,
                 X: np.ndarray,
                 shape: tuple,
                 loss_grads: Callable,
                 iterations: int=1000,
                 callback: Callable=None):
        """
        Reduce the loss geanerated by X.

        Args:
            X: the input value to adjust to minimize loss
            shape: the shape to coerce X to
            loss_grads: a callable method that returns loss and gradients
                        given some input
            iterations: the number of iterations of optimization to perform
            callback: an optional callback method to receive image updates

        Returns: an optimized X about the loss and gradients given
        """

        def loss_and_gradients(X):
            """Calculate the loss and gradients with appropriate reshaping."""
            loss, gradients = loss_grads([X.reshape(shape)])
            return loss, gradients.flatten().astype('float64')

        # assign the custom method to self for loss / gradient calculation
        self.loss_and_gradients = loss_and_gradients

        for i in range(iterations):
            # pass X through an iteration of LBFGS
            X, min_val, info = fmin_l_bfgs_b(self.loss, X,
                                             fprime=self.gradients,
                                             maxfun=self.max_evaluations)
            # pass the values to the callback if any
            if callable(callback):
                callback(X.reshape(shape), i)

        return X


__all__ = ['L_BFGS']
