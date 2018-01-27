"""Image utility methods for the project."""
from PIL import Image
import numpy as np


# the BGR means from the ImageNet database
IMAGENET_MEANS = np.array([103.939, 116.779, 123.68])


<<<<<<< HEAD
def load_image(img_path: str, size: tuple=None) -> Image:
=======
def load_image(image_path: str, size: tuple=None) -> Image:
>>>>>>> 079fe4d2e66f75c64f6a51ba537692dd90d8654b
	"""
	Load and return an image at a given path.

	Args:
<<<<<<< HEAD
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
=======
		image_path: the path of the image to load
		size: a tuple of (width, height) to resize the image (default None)

	Returns:
		the image loaded from the given path
	"""
	# load the image from the path
	image = Image.open(image_path)
	# if a size is provided
	if size is not None:
		# apply the size to the image
		image = image.resize(size)
	return image


def image_to_matrix(image: Image, dtype: np.dtype=np.float32) -> np.ndarray:
	"""
	Convert the input image to a 4D NumPy matrix.

	Args:
		image: the image to convert to a matrix

	Returns:
		a [1, height, width, channel] matrix representation of the image
	"""
	# convert the image to a numpy matrix of [height, width, channel]
	image = np.asarray(image, dtype=dtype)
	# expand the image into the 4th dimension.
	# i.e. [frame, height, wight, channel] but with just a single frame of this
	# image.
	image = np.expand_dims(image, axis=0)

	return image


def matrix_to_image(image: np.ndarray, channel_range: tuple=(0, 255)) -> Image:
>>>>>>> 079fe4d2e66f75c64f6a51ba537692dd90d8654b
	"""
	Convert the input matrix to an image.

	Args:
<<<<<<< HEAD
		img: the matrix of shape (1, w, h, d) to convert

	Returns: an image from the pixels in the img array
	"""
	# clip the image into rgb pixel values instead of [0, 1] floats
	rgb_pixels = np.clip(img, 0, 255).astype('uint8')
	return Image.fromarray(rgb_pixels)


def normalize(img: np.ndarray):
=======
		image: the matrix of shape [height, width, channel] to convert
		channel_range: the range to clip the channel values to (inclusive)

	Returns:
		an image from the pixels in the image array
	"""
	# clip the values in the image to the boundary [0, 255]. This is the
	# legal range for channel values. Image uses a method called 'to bytes'
	# to compress the input array into a simpler binary representation for
	# graphics processing. As such, convert the type to a single byte to
	# satisfy this constraint.
	image = np.clip(image, *channel_range).astype('uint8')
	return Image.fromarray(image)


def normalize(image: np.ndarray, inplace: bool=False):
>>>>>>> 079fe4d2e66f75c64f6a51ba537692dd90d8654b
    """
    Normalize an image by a set of means.

    Args:
<<<<<<< HEAD
        img: the image to normalize (assuming ML frame shape)

    Returns: img after normalizing its RGB scale by the means
    """
    # flip image from RGB, to BGR
    img = img[:, :, :, ::-1]
    # vector subtract the means from ImageNet to the image
    img[:, :, :, np.arange(IMAGENET_MEANS.shape[0])] -= IMAGENET_MEANS

    return img


def denormalize(img: np.ndarray):
=======
        image: the image to normalize assuming shape [1, height, width, channel]
        inplace: whether to perform the operation of the array in-place

    Returns:
    	image after flipping from RGB to BGR and subtracting ImageNet means
    """
    # validate the shape of the image
    assert image.shape[3] == IMAGENET_MEANS.shape[0]
    # if in-place is enabled, copy the array
    if not inplace:
    	image = image.copy()
    # flip image from RGB, to BGR
    image = image[:, :, :, ::-1]
    # vector subtract the means from ImageNet to the image
    image[:, :, :, np.arange(IMAGENET_MEANS.shape[0])] -= IMAGENET_MEANS

    return image


def denormalize(image: np.ndarray, inplace: bool=False):
>>>>>>> 079fe4d2e66f75c64f6a51ba537692dd90d8654b
    """
    De-normalize an image by a set of means.

    Args:
<<<<<<< HEAD
        img: the image to normalize (assuming just the image shape)

    Returns: img after normalizing its RGB scale by the means
    """
    # vector add the means from ImageNet to the image
    img[:, :, np.arange(IMAGENET_MEANS.shape[0])] += IMAGENET_MEANS
    # flip image from BGR, to RGB
    img = img[:, :, ::-1]

    return img
=======
        image: the image to normalize (assuming standard image shape in BGR)

    Returns:
    	image after flipping from BGR to RGB and adding ImageNet means
    """
    # validate the shape of the image
    assert image.shape[2] == IMAGENET_MEANS.shape[0]
    # if in-place is enabled, copy the array
    if not inplace:
    	image = image.copy()
    # vector add the means from ImageNet to the image
    image[:, :, np.arange(IMAGENET_MEANS.shape[0])] += IMAGENET_MEANS
    # flip image from BGR, to RGB
    image = image[:, :, ::-1]

    return image
>>>>>>> 079fe4d2e66f75c64f6a51ba537692dd90d8654b


# explicitly specify the public API of the module
__all__ = [
	'load_image',
	'image_to_matrix',
	'matrix_to_image',
	'normalize',
	'denormalize'
]
