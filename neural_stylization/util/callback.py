"""A method for building a rich callback for optimizers."""
import os
from glob import glob
from IPython import display
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from .img_util import denormalize


class Callback:
    """A callback for displaying and saving images."""

    def __init__(self,
        out_dir: str=None,
        extension='.png',
        plot=True,
        clear_display=True,
    ):
        """
        Initialize a new callback.

        Args:
            out_dir: the name of the artwork directory to store frames in
            extension: the file-type to save the images as
            plot: whether to plot the images
            clear_display: whether to clear the IPython display for new plots

        Returns:
            None

        """
        self.out_dir = out_dir
        self.extension = extension
        self.plot = plot
        self.clear_display = clear_display
        # make the directory if it doesn't exist
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        # delete contents of the directory in case
        # it already existed with stuff in it
        for file in glob(f'{self.out_dir}/{self.extension}'):
            os.remove(file)

    def __call__(self, image, i):
        """
        De-normalize an iteration of optimization to display.

        Args:
            image: the image to de-normalize and display
            i: the iteration of optimization

        Returns:
            None

        """
        image = np.clip(denormalize(image[0]), 0, 255).astype('uint8')
        if self.out_dir is not None:
            io.imsave('{}/{}.jpg'.format(self.out_dir, i), image)
        if self.plot:
            if self.clear_display:
                display.clear_output(wait=True)
            ax = plt.imshow(image)
            ax.axes.xaxis.set_major_locator(plt.NullLocator())
            ax.axes.yaxis.set_major_locator(plt.NullLocator())
            plt.show()


# explicitly define the outward facing API of this module
__all__ = [Callback.__name__]
