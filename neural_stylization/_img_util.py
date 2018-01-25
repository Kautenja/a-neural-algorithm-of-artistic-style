"""Image utility methods for the project."""
from PIL import Image
import numpy as np


# the BGR means from the ImageNet database
IMAGENET_MEANS = np.array([103.939, 116.779, 123.68])


def load_image(img_path: str, size: tuple=None) -> Image:
	"""
	Load and return an image at a given path.

	Args:
		img_path: the path of the image to load
		size: a tuple of (w,h) to resize the image (default None)

	Returns: the image loaded from the given path
	"""
	# load the image from the path
	img = Image.open(img_path)
	# if a size is provided
	if size is not None:
		# apply the size to the image
		img = img.resize(size)
	return img


def image_to_matrix(img: Image, dtype: np.dtype=np.float32) -> np.ndarray:
	"""
	Convert the input image to a matrix.

	Args:
		img: the image to convert to a matrix

	Returns: a (W x H x d) matrix representing the image
	"""
	# convert the image to a numpy matrix of (W x H x d) where W and H are the
	# dimensions and d is the color depth (1 for grayscale, 3 for color (RGB))
	img = np.asarray(img, dtype=dtype)
	img = np.expand_dims(img, axis=0)

	return img


def matrix_to_image(img: np.ndarray) -> Image:
	"""
	Convert the input matrix to an image.

	Args:
		img: the matrix of shape (1, w, h, d) to convert

	Returns: an image from the pixels in the img array
	"""
	# clip the image into rgb pixel values instead of [0, 1] floats
	rgb_pixels = np.clip(img, 0, 255).astype('uint8')
	return Image.fromarray(rgb_pixels)


def normalize(img: np.ndarray):
    """
    Normalize an image by a set of means.

    Args:
        img: the image to normalize (assuming ML frame shape)

    Returns: img after normalizing its RGB scale by the means
    """
    # flip image from RGB, to BGR
    img = img[:, :, :, ::-1]
    # vector subtract the means from ImageNet to the image
    img[:, :, :, np.arange(IMAGENET_MEANS.shape[0])] -= IMAGENET_MEANS

    return img


def denormalize(img: np.ndarray):
    """
    De-normalize an image by a set of means.

    Args:
        img: the image to normalize (assuming just the image shape)

    Returns: img after normalizing its RGB scale by the means
    """
    # vector add the means from ImageNet to the image
    img[:, :, np.arange(IMAGENET_MEANS.shape[0])] += IMAGENET_MEANS
    # flip image from BGR, to RGB
    img = img[:, :, ::-1]

    return img


# explicitly specify the public API of the module
__all__ = [
	'load_image',
	'image_to_matrix',
	'matrix_to_image',
	'normalize',
	'denormalize'
]
