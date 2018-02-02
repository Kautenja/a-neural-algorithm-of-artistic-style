"""Methods to synthesize images based on content and style loss."""
from typing import List
import numpy as np
from keras import backend as K
from .vgg19 import VGG_19
from .loss_functions import (
    content_loss,
    style_loss,
    total_variation_loss
)
from .jupyter_plot import JupyterPlot
from ._img_util import (
    load_image,
    image_to_matrix,
    normalize,
    denormalize,
    matrix_to_image
)


class Stylizer(object):
    """An implementation of "A Neural Algorithm of Artistic Style"."""

    def __init__(self,
                 content_layer_name: str='block4_conv2',
                 style_layers_names: List[str]=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'],
                 learning_rate=1e-4,
                 iterations: int=1000,
                 content_weight: float=1,
                 style_weight: float=1000,
                 total_variation_weight: float=0.1) -> None:
        """
        Initialize a new "Neural Algorithm of Artistic Style".

        TODO:
        Args:
            content_layer_name:
            style_layers_names:
            learning_rate:
            iterations:
            content_weight:
            style_weight:
            total_variation_weight:

        Returns:
            None

        """
        # TODO: type and error check
        self.content_layer_name = content_layer_name
        self.style_layers_names = style_layers_names
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.total_variation_weight = total_variation_weight


    @property
    def content_style_scale(self) -> float:
        """Return alpha / beta i.e. (content_weight / style_weight)."""
        return self.content_weight / self.style_weight

    def load_and_process(self, img_path: str, dimensions: tuple) -> np.ndarray:
        """
        Load and process the image at the given path with given dimensions.

        Args:
            img_path: the path of the image to load
            dimensions: the dimensions to apply to the image

        Returns: a 4D NumPy array of the image

        """
        # load the image from disk as a binary image file
        img = load_image(img_path, dimensions)
        # convert the image to a 4D matrix
        img = image_to_matrix(img)
        # normalize the BGR channels by the ImageNet means
        img = normalize(img)

        return img

    def create_input_tensor(self, content: np.ndarray, style: np.ndarray):
        """
        Create an input tensor given the matrices of content and style.

        Args:
            content: the 4D content matrix
            style: the 4D style matrix

        Returns:
            a 4D tensor of content, style, and placeholder canvas

        """
        # load the content image into Keras as a constant, it never changes
        content_tensor = K.constant(content, name='Content')
        # load the style image into Keras as a constant, it never changes
        style_tensor = K.constant(style, name='Style')
        # create a placeholder for the trained image, this variable changes
        canvas_tensor = K.placeholder(content.shape, name='Canvas')
        # create the list of tensors to combine into one THE ORDER MATTERS
        input_tensor = [content_tensor, style_tensor, canvas_tensor]
        # combine the content, style, and canvas tensors along the frame
        # axis (0) into a 4D tensor of shape [2, height, width, channels]
        input_tensor = K.concatenate(input_tensor, axis=0)

        return input_tensor


# build the model with the input tensor of content, style, and canvas
model = VGG_19(include_top=False,
               input_tensor=input_tensor,
               pooling='avg')

# LOSSES
# initialize a variable to store the loss into
loss = K.variable(0.0, name='Loss')

# CONTENT LOSS
# extract the content layer tensor for optimizing content loss
content_layer_output = model[content_layer_name]
# calculate the loss between the output of the layer on the
# content (0) and the canvas (2)
loss += content_weight * content_loss(content_layer_output[0], content_layer_output[2])

# STYLE LOSS
# iterate over the list of all the layers that we want to include
for style_layer_name in style_layers_names:
    # extract the layer's out that we have interest in for reconstruction
    style_layer_output = model[style_layer_name]
    # calculate the loss between the output of the layer on the
    # style (1) and the canvas (2).
    style_loss_layer = style_loss(style_layer_output[1], style_layer_output[2])
    # Apply the lazy w_l factor of dividing by the size of the styling
    # layer list. multiply the style weight in here
    loss += style_loss_layer * style_weight / len(style_layers_names)

# TOTAL VARIATION LOSS
# add the total variation loss based on the euclidean distance between shifted points
loss += total_variation_weight * total_variation_loss(canvas)

# GRADIENTS
# calculate the gradients of the input image with respect to
# the loss. i.e. backpropagate the loss through the network
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
iterate = K.function([canvas], [loss, grads])

# generate random noise
noise = normalize(np.random.uniform(0, 128, content.shape))
# create a new interactive plot to visualize the loss in realtime
plot = JupyterPlot(title='Total Loss by Iteration', xlabel='Iteration', ylabel='Loss')
# perform the specified iterations of gradient descent
for i in range(iterations):
    # pass the noise the canvas tensor generating the loss
    # and gradients as a tuple
    loss_i, grads_i = iterate([noise])
    # move the noise based on the gradients and learning rate
    noise -= learning_rate * grads_i
    # update the plot with the loss for this iteration
    plot(loss_i)
#     display.clear_output(wait=True)
#     display.display(matrix_to_image(denormalize(noise[0])))
#     display.display(i)

# clear out all the keras variables from the GPU/CPU
K.clear_session()

# denormalize the image to add the mean values
# of the network back and flip the channels back
# to RGB from BGR. convert this RGB matrix to
# an image we can look at
matrix_to_image(denormalize(noise[0]))
