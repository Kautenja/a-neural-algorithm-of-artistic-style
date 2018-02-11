"""A method for building a rich callback for optimizers."""


def build_callback(out_dir: str):
    """
    Build a callback method for the given artwork.

    Args:
        out_dir: the name of the artwork directory to store frames in

    Returns:
        a callable method for de-normalizing, displaying, and saving frames

    """
    # make the directory if it doesn't exist
    import os
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # delete contents of the directory in case
    # it already existed with stuff in it
    from glob import glob
    for file in glob('{}/*.png'.format(out_dir)):
        os.remove(file)

    from .img_util import denormalize
    from .img_util import matrix_to_image
    from IPython import display

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
        image = matrix_to_image(denormalize(image[0]))
        # write the image to disk in the appropriate spot
        image.save('{}/{}.png'.format(out_dir, i))
        # display the image on the IPython front end
        display.display(image)

    return denormalize_and_display
