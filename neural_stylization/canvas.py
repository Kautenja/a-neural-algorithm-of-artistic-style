"""An abstraction of a canvas for the neural algorithm of artistic style."""
import numpy as np
from keras import backend as K
from PIL import Image
from ._img_util import load_image, normalize, image_to_matrix, matrix_to_image


class Canvas(object):
    """A canvas for the neural algorithm of artistic style."""

    def __init__(self,
                 content_path: str,
                 style_path: str,
                 dimensions: tuple=None) -> None:
        """
        A canvas determining where to get content & style and where to save.

        Args:
            content_path: the path to the image to use as content
            style_path: the path to the image to use for style
            dimensions: a tuple of (width, height) determining optional dims

        Returns: None
        """
        self.content_path = content_path
        self.style_path = style_path

        if dimensions is None:
            # load the content image
            self.content_image = load_image(content_path)
            self.content = normalize(image_to_matrix(self.content_image))
            # store the dimensions of the canvas based on the content size
            self.height, self.width = self.content.shape[1], self.content.shape[2]
        else:
            # set the dimensions of the canvas based on the parameter
            self.width, self.height = dimensions
            # load the content image with the custom size
            self.content_image = load_image(content_path, (self.width, self.height))
            self.content = normalize(image_to_matrix(self.content_image))

        # load the style image (using the dimensions of the content)
        self.style_image = load_image(style_path, (self.width, self.height))
        self.style = normalize(image_to_matrix(self.style_image))

        # load the variables into tensorflow
        self.content = K.variable(self.content)
        self.style = K.variable(self.style)
        self.output = K.placeholder((1, self.height, self.width, 3))
        data = [self.content, self.style, self.output]

        # the input tensor produced by all three images combined
        self.input_tensor = K.concatenate(data, axis=0)

    def __repr__(self) -> str:
        """Return a debugging representation of self."""
        template = "{}(content_path='{}', style_path='{}', dimensions={})"
        return template.format(*[
            self.__class__.__name__,
            self.content_path,
            self.style_path,
            (self.width, self.height)
        ])

    def __str__(self) -> str:
        """Return a human friendly string of self."""
        return f'Canvas of ({self.width}, {self.height})'

    @property
    def shape(self) -> tuple:
    	"""Return the shape of this image."""
    	return (1, self.height, self.width, 3)

    @property
    def random_noise(self) -> np.ndarray:
        """Return an image of noise the same size as this canvas."""
        return np.random.uniform(0, 255, self.shape) - 128.0

    @property
    def random_noise_image(self) -> Image:
        """Return a decoded image of random noise in the size of this canvas."""
        noise = self.random_noise.reshape((self.height, self.width, 3))
        return matrix_to_image(noise)


# explicitly export the public API of this module
__all__ = ['Canvas']
