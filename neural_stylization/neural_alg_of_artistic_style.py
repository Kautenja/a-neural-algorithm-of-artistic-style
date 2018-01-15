"A Neural Algorithm of Artistic Style."
import time
from numpy.random import uniform
from keras import backend
from scipy.optimize import fmin_l_bfgs_b
from .img_util import load_image
from .img_util import image_to_matrix, matrix_to_image
from .img_util import normalize, denormalize
from .vgg19 import VGG_19
from .loss_functions import gram_matrix
from .loss_functions import content_loss, style_loss, total_variation_loss
from .loss_functions import Evaluator


class Canvas(object):
    """A canvas for the neural algorithm of artistic style."""

    # the template for the repr method for this object
    REPR = "{}(content_path='{}', style_path='{}', output_path='{}')"

    def __init__(self, content_path: str, style_path: str, output_path: str):
        """
        A canvas determining where to get content & style and where to save.

        Args:
            content_path: the path to the image to use as content
            style_path: the path to the image to use for style
            output_path: the path to output images to

        Returns: None
        """
        self.content_path = content_path
        self.style_path = style_path
        self.output_path = output_path

        # load the content image
        self.content_image = load_image(content_path)
        self.content = normalize(image_to_matrix(self.content_image))

        # store the height of the canvas based on the content size
        self.height, self.width = self.content.shape[1], self.content.shape[2]

        # load the style image (using the dimensions of the content)
        self.style_image = load_image(style_path, (self.width, self.height))
        self.style = normalize(image_to_matrix(self.style_image))

        # load the variables into tensorflow
        self.content = backend.variable(self.content)
        self.style = backend.variable(self.style)
        self.output = backend.placeholder((1, self.height, self.width, 3))

    def __repr__(self):
        """Return a debugging representation of self."""
        return self.REPR.format(self.__class__.__name__,
                                self.content_path,
                                self.style_path,
                                self.output_path)

    def __str__(self):
        """Return a human friendly string of self."""
        return f'Canvas of ({self.width}, {self.height})'

    @property
    def input_tensor(self):
        """Return an input tensor based on the data in this canvas."""
        # concatentate the images into a single tensor
        return backend.concatenate([self.content, self.style, self.output],
                                   axis=0)

    @property
    def random_noise(self):
        """Return an image of noise the same size as this canvas."""
        return uniform(0, 255, (1, self.height, self.width, 3)) - 128.0


# content layer configurations by paper that references them
CONTENT_LAYER = {
    'Gatys et al. (2015)': 'block4_conv2',
    'Johnson et al. (2016)': 'block2_conv2'
}

# style layer configurations by paper that references them
STYLE_LAYERS = {
    'Gatys et al. (2015)': ['block1_conv1', 'block2_conv1', 'block3_conv1',
                            'block4_conv1', 'block5_conv1'],
    'Johnson et al. (2016)': ['block1_conv2', 'block2_conv2', 'block3_conv3',
                              'block4_conv3', 'block5_conv3']
}


class NeuralAlgorithmOfArtisticStyle(object):
    """A Neural Algorithm of Artistic Style."""

    # the template for the repr method for this object
    REPR = "{}(content_weight='{}', style_weight='{}', variation_weight='{}', content_layer={}, style_layers={})"

    def __init__(self,
                 content_weight: float=0.025,
                 style_weight: float=5.0,
                 variation_weight: float=1.0,
                 content_layer: str=CONTENT_LAYER['Gatys et al. (2015)'],
                 style_layers: list=STYLE_LAYERS['Gatys et al. (2015)']):
        """
        Initialize a new neural algorithm of artistic style.

        Args:
            content_weight: determines the prevalence of the content in the
                output
            style_weight: determines the prevalence of the style in the output
            variation_weight: determines the amount of noise in the output
            content_layer: the layer of the CNN to use for the content
            style_layers: the layers of the CNN to use for the stylization.

        Returns: None
        """
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.variation_weight = variation_weight
        self.content_layer = content_layer
        self.style_layers = style_layers

    def __repr__(self):
        """Return a debugging representation of self."""
        return self.REPR.format(self.__class__.__name__,
                                self.content_weight,
                                self.style_weight,
                                self.variation_weight,
                                self.content_layer,
                                self.style_layers)

    def __call__(self, canvas: Canvas, iterations: int=1) -> Canvas:
        """
        Optimize the loss between a content and style image on a canvas.

        Args:
            canvas: the canvas to draw inspiration from and to draw to

        Returns: a mutated canvas with the optimized image
        """
        # instantiate a new VGG_19 model. include_top tells the model to
        # not include the fully connected layers at the end (not interested
        # in classification). We use the input tensor from the canvas to define
        # the input dimension of the network.
        model = VGG_19(input_tensor=canvas.input_tensor, include_top=False)
        # the various layers in the model indexed by name
        layers = dict([(layer.name, layer.output) for layer in model.layers])

        # the loss variable for accumulating the loss
        loss = backend.variable(0.0)

        # Calculate the content loss
        layer_features = layers[self.content_layer]
        content_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        # multiply the content loss by the content weight
        loss += self.content_weight * content_loss(content_image_features,
                                                   combination_features)

        # iterate over the style layers to stylize the image
        for layer_name in self.style_layers:
            layer_features = layers[layer_name]
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_features, combination_features,
                            canvas.width, canvas.height)
            loss += (self.style_weight / len(self.style_layers)) * sl

        # apply the weighted variation loss for noise
        loss += self.variation_weight * total_variation_loss(canvas)

        # define the gradients of the loss function and the canvas output
        grads = backend.gradients(loss, canvas.output)

        # define a function to add the gradients to the loss
        outputs = [loss]
        outputs += grads
        f_outputs = backend.function([canvas.output], outputs)

        def eval_loss_and_grads(x):
            x = x.reshape((1, canvas.height, canvas.width, 3))
            outs = f_outputs([x])
            loss_value = outs[0]
            grad_values = outs[1].flatten().astype('float64')
            return loss_value, grad_values

        evaluator = Evaluator(eval_loss_and_grads)

        return self._optimize(evaluator, canvas, iterations)

    def _optimize(self, evaluator: Evaluator, canvas: Canvas, iterations: int):
        """
        """
        x = canvas.random_noise

        for i in range(iterations):
            print('Start of iteration', i)
            start_time = time.time()
            x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                             fprime=evaluator.grads, maxfun=20)
            print('Current loss value:', min_val)
            end_time = time.time()
            print('Iteration %d completed in %ds' % (i, end_time - start_time))

        # reshape, denormalize, and convert to an image object
        x = x.reshape((canvas.height, canvas.width, 3))
        x = denormalize(x)
        x = matrix_to_image(x)

        return x


# explicitly specify the public API of the module
__all__ = ['NeuralAlgorithmOfArtisticStyle']
