"A Neural Algorithm of Artistic Style."
import numpy as np
from .vgg19 import VGG_19


class NeuralAlgorithmOfArtisticStyle(object):
    """A Neural Algorithm of Artistic Style."""

    def __init__(self,
                 content_weight: float=0.025,
                 style_weight: float=5.0,
                 variation_weight: float=1.0):
        """
        Initialize a new neural algorithm of artistic style.

        Args:
            content_weight: determines the prevalence of the content in the
                output
            style_weight: determines the prevalence of the style in the output
            variation_weight: determines the amount of noise in the output
                (default 1.0)
        """
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.variation_weight = variation_weight


# explicitly specify the public API of the module
__all__ = ['NeuralAlgorithmOfArtisticStyle']
