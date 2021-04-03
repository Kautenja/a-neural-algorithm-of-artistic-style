"""A method for building a rich callback for optimizers."""
import os
from glob import glob
from IPython import display
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from .img_util import denormalize


def build_callback(out_dir: str=None):
    """
    Build a callback method for the given artwork.

    Args:
        out_dir: the name of the artwork directory to store frames in

    Returns:
        a callable method for de-normalizing, displaying, and saving frames

    """
    # make the directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # delete contents of the directory in case
    # it already existed with stuff in it
    for file in glob('{}/*.png'.format(out_dir)):
        os.remove(file)

    def denormalize_and_display(image, i) -> None:
        """
        De-normalize an iteration of optimization to display.

        Args:
            image: the image to de-normalize and display
            i: the iteration of optimization

        Returns:
            None

        """
        # clear the existing output
        display.clear_output(wait=True)
        # de-normalize the image and convert to binary
        image = np.clip(denormalize(image[0]), 0, 255).astype('uint8')
        # write the image to disk in the appropriate spot
        if out_dir is not None:
            io.imsave('{}/{}.png'.format(out_dir, i), image)
        # display the image on the IPython front end
        ax = plt.imshow(image)
        ax.axes.xaxis.set_major_locator(plt.NullLocator())
        ax.axes.yaxis.set_major_locator(plt.NullLocator())
        plt.show()

    return denormalize_and_display


# explicitly define the outward facing API of this module
__all__ = [build_callback.__name__]
