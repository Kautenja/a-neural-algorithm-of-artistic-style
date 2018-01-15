"""Image utility methods for the project."""
from PIL import Image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input


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
	# pass the array through the preprocessing method provided by keras. this
	# adjusts the image to account for the ImageNet dataset.
	return preprocess_input(img)


# explicitly specify the public API of the module
__all__ = ['load_image', 'image_to_matrix']
