"""A mechanism for transferring style of art to content."""
from typing import Callable, List
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg19 import VGG19
from .util.img_util import normalize
from .util.img_util import denormalize
from .loss_functions import content_loss, style_loss, total_variation_loss


# the template for the class's __repr__ method
TEMPLATE = """{}(
    content_layer_name={},
    content_weight={},
    style_layer_names={},
    style_layer_weights={},
    style_weight={},
    total_variation_weight={}
)""".lstrip()


class Stylizer(object):
    """An algorithm for stylizing images based on artwork."""

    def __init__(self,
        content_layer_name: str='block4_conv2',
        content_weight: float=1.0,
        style_layer_names: List[str]=[
            'block1_conv1',
            'block2_conv1',
            'block3_conv1',
            'block4_conv1',
            'block5_conv1'
        ],
        style_layer_weights: List[float]=None,
        style_weight: float=10000.0,
        total_variation_weight: float=0.0
    ) -> None:
        """
        Initialize a new neural stylization algorithm.

        Args:
            content_layer_name: the name of the layer to extract content from
            content_weight: the weight, alpha, to attribute to content loss
            style_layer_names: the names of the layers to extract style from
            style_weight: the weight, beta, to attribute to style loss
            style_layer_weights: the set of weights to apply to the individual
                losses from each style layer. If None, the default is to take
                the average, i.e. divide each by len(style_layer_names).
            total_variation_weight: the amount of total variation de-noising
                to apply to the synthetic images

        Returns:
            None

        """
        # get the names of the layers from the model to error check
        layer_names = [l.name for l in VGG19(include_top=False).layers]

        # type and value check: content_layer_name
        if not isinstance(content_layer_name, str):
            raise TypeError('`content_layer_name` must be of type: str')
        if content_layer_name not in layer_names:
            raise ValueError(
                '`content_layer_name` must be a layer name in VGG19'
            )
        self.content_layer_name = content_layer_name

        # type and value check: content_weight
        if not isinstance(content_weight, (int, float)):
            raise TypeError('`content_weight` must be of type: int or float')
        if content_weight < 0:
            raise ValueError('`content_weight` must be >= 0')
        self.content_weight = content_weight

        # type and value check: content_layer_name
        if not isinstance(style_layer_names, list):
            raise TypeError('`style_layer_names` must be of type: list')
        if not all(layer in layer_names for layer in style_layer_names):
            raise ValueError(
                '`style_layer_names` must be a list of layer names in VGG19'
            )
        self.style_layer_names = style_layer_names

        # type and value check: style_layer_weights
        if style_layer_weights is None:
            # initialize style layer weights as an average between them.
            total = len(style_layer_names)
            style_layer_weights = total * [1.0 / total]
        else:
            if not isinstance(style_layer_weights, list):
                raise TypeError(
                    '`style_layer_weights` must be of type: None or list'
                )
            if not all(isinstance(w, (float, int)) for w in style_layer_weights):
                raise ValueError(
                    '`style_layer_weights` must be a list of numbers or None'
                )
        self.style_layer_weights = style_layer_weights

        # type and value check: style_weight
        if not isinstance(style_weight, (int, float)):
            raise TypeError('`style_weight` must be of type: int or float')
        if style_weight < 0:
            raise ValueError('`style_weight` must be >= 0')
        self.style_weight = style_weight

        # type and value check: total_variation_weight
        if not isinstance(total_variation_weight, (int, float)):
            raise TypeError(
                '`total_variation_weight` must be of type: int or float'
            )
        if total_variation_weight < 0:
            raise ValueError('`total_variation_weight` must be >= 0')
        self.total_variation_weight = total_variation_weight

        # disable eager mode for this operation
        tf.compat.v1.disable_eager_execution()

    def __repr__(self) -> str:
        """Return an executable string representation of this object."""
        return TEMPLATE.format(*[
            self.__class__.__name__,
            repr(self.content_layer_name),
            self.content_weight,
            self.style_layer_names,
            self.style_layer_weights,
            self.style_weight,
            self.total_variation_weight
        ])

    @property
    def content_style_ratio(self) -> float:
        """Return the ratio of content weight to style weight."""
        return self.content_weight / self.style_weight

    def _build_model(self, content: np.ndarray, style: np.ndarray) -> tuple:
        """
        Build a synthesis model with the given content and style.

        Args:
            content: the content to fuse the artwork into
            style: the artwork to get the style from

        Returns:
            a tuple of:
            -   the constructed VGG19 model from the input images
            -   the canvas tensor for the synthesized image

        """
        # load the content image into Keras as a constant, it never changes
        content_tensor = K.constant(content, name='Content')
        # load the style image into Keras as a constant, it never changes
        style_tensor = K.constant(style, name='Style')
        # create a placeholder for the trained image, this variable changes
        canvas = K.placeholder(content.shape, name='Canvas')
        # combine the content, style, and canvas tensors along the frame
        # axis (0) into a 4D tensor of shape [3, height, width, channels]
        tensor = K.concatenate([content_tensor, style_tensor, canvas], axis=0)
        # build the model with the input tensor of content, style, and canvas
        model = VGG19(include_top=False, input_tensor=tensor, pooling='avg')

        return model, canvas

    def _build_loss_grads(self, model, canvas: 'Tensor') -> Callable:
        """
        Build the optimization methods for stylizing the image from a model.

        Args:
            model: the model to extract layers from
            canvas: the input to the model thats being mutated

        Returns:
            a function to calculate loss and gradients from an input X

        """
        # initialize a variable to store the loss into
        loss = K.variable(0.0, name='Loss')

        # CONTENT LOSS
        # extract the content layer tensor for optimizing content loss
        content_layer_output = model.get_layer(self.content_layer_name).output
        # calculate the loss between the output of the layer on the
        # content (0) and the canvas (2)
        cl = content_loss(content_layer_output[0],
                          content_layer_output[2])
        loss = loss + self.content_weight * cl

        # STYLE LOSS
        sl = K.variable(0.0)
        # iterate over the list of all the layers that we want to include
        for style_layer_name, layer_weight in zip(self.style_layer_names,
                                                  self.style_layer_weights):
            # extract the layer out that we have interest in
            style_layer_output = model.get_layer(style_layer_name).output
            # calculate the loss between the output of the layer on the
            # style (1) and the canvas (2).
            sl = sl + layer_weight * style_loss(style_layer_output[1],
                                                style_layer_output[2])
        # apply the style weight to style loss and add it to the total loss
        loss = loss + self.style_weight * sl

        # TOTAL VARIATION LOSS
        # Gatys et al. don't use the total variation de-noising in their paper
        # (or at least they never mention it) so the weight is 0.0 by
        # default, but can be applied if desired
        loss = loss + self.total_variation_weight * total_variation_loss(canvas)

        # GRADIENTS
        # calculate the gradients of the input image with respect to
        # the loss. i.e. back-propagate the loss through the network
        # to the input layer (only the canvas though)
        grads = K.gradients(loss, canvas)[0]

        # generate the iteration function for gradient descent optimization
        # Args:
        #     noise: the input to the noise placeholder in the model
        #         this effectively takes a the white noise image being
        #         optimized and passes it forward and backward through
        #         the model collecting the loss and gradient along the
        #         way
        #
        # Returns:
        #     a tuple of (loss, gradients)
        #     -   loss: the content loss between the content, style image
        #         and the white noise
        #     -   gradients: the gradients of the inputs with respect
        #         to the loss
        return K.function([canvas], [loss, grads])

    def __call__(self, content: np.ndarray, style: np.ndarray, optimize: Callable,
            iterations: int = 10,
            initialization_strat: str = 'noise',
            noise_range: tuple = (0, 1),
            callback: Callable = None
        ):
        """
        Stylize the given content image with the give style image.

        Args:
            content: the content image to use
            style: the style image to use
            optimize: the black-box optimizer to use. This is a callable
                method conforming to the API for optimizers
            iterations: the number of optimization iterations to perform
            image_size: the custom size to load images with, if any. When
                set to None, the size of the content image will be used
            initialization_strat: the way to initialize the canvas for the
                style transfer. Can be one of:
                -   'noise': initialize the canvas as random noise with the
                    given noise range for sampling pixels
                -   'content': initialize the canvas as the content image
                -   'style': initialize the canvas as the style image
            noise_range: the custom range for initializing random noise. This
                option is only used when `initialization_strat` is 'noise'.
            callback: the optional callback method for optimizer iterations

        Returns:
            the image as a result of blending content with style

        """
        # normalize the input data
        content = normalize(content[None, ...].astype('float'))
        style = normalize(style[None, ...].astype('float'))

        # disable eager mode for this operation
        tf.compat.v1.disable_eager_execution()
        # build the inputs tensor from the images
        model, canvas = self._build_model(content, style)
        # build the iteration function
        loss_grads = self._build_loss_grads(model, canvas)

        # setup the initial image for the optimizer
        if initialization_strat == 'noise':
            # generate white noise in the shape of the canvas
            initial = np.random.uniform(*noise_range, size=canvas.shape)
        elif initialization_strat == 'content':
            # copy the content as the initial image
            initial = content.copy()
        elif initialization_strat == 'style':
            # copy the style as the initial image
            initial = style.copy()
        else:
            raise ValueError(
                "`initialization_strat` must be one of: ",
                " 'noise', 'content', 'style' "
            )

        # optimize the initial image into a synthetic painting. Name all args
        # by keyword to help catch erroneous optimize callables.
        image = optimize(
            X=initial,
            shape=canvas.shape,
            loss_grads=loss_grads,
            iterations=iterations,
            callback=callback
        )

        # clear the Keras session (this removes the variables from memory).
        K.clear_session()

        # reshape the image in case the optimizer did something funky with the
        # shape. `denormalize` expects a vector of shape [h, w, c], but
        # canvas has an additional dimension for frame, [frame, h, w, c]. Ss
        # such, take the first item along the frame axis
        image = image.reshape(canvas.shape)[0]
        # denormalize the image about the ImageNet means. this will invert the
        # channel dimension turning image from BGR to RGB
        return denormalize(image).astype('uint8')


# explicitly define the outward facing API of this module
__all__ = [Stylizer.__name__]
