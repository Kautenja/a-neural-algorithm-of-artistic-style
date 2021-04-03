"""Image utility methods for the project."""
import numpy as np


# the BGR means from the ImageNet database
IMAGENET_MEANS = np.array([103.939, 116.779, 123.68])


def normalize(image: np.ndarray, inplace: bool=False) -> np.ndarray:
    """
    Normalize an image by a set of means.

    Args:
        image: the image to normalize assuming shape [1, H, W, channels]
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


def denormalize(image: np.ndarray, inplace: bool=False) -> np.ndarray:
    """
    De-normalize an image by a set of means.

    Args:
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


# explicitly specify the public API of the module
__all__ = [normalize.__name__, denormalize.__name__]
