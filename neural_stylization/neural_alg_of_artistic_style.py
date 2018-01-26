"""A Neural Algorithm of Artistic Style."""
from tqdm import tqdm, tqdm_notebook
from PIL import Image
from keras import backend
# tensorflow uses some Garbage ad-hoc module build scheme that wont let you
# import from submodules in the package the standard Pythonic way. As such
# import it first so it's ad-hoc garbage to be imported from.
import tensorflow as tf
# from scipy.optimize import fmin_l_bfgs_b as optimize
# from scipy.optimize import fmin_tnc as optimize
from ._img_util import denormalize, matrix_to_image
from .vgg19 import VGG_19
from .loss_functions import content_loss, style_loss, total_variation_loss
from .loss_functions import Evaluator
from .canvas import Canvas


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

# the template for the repr method for this object
REPR = """{}(
    content_weight={},
    style_weight={},
    variation_weight={},
    content_layer='{}',
    style_layers={},
    is_notebook={}
)
""".lstrip()


class NeuralAlgorithmOfArtisticStyle(object):
    """A Neural Algorithm of Artistic Style."""

    def __init__(self,
                 content_weight: float=0.025,
                 style_weight: float=5.0,
                 variation_weight: float=1.0,
                 content_layer: str=CONTENT_LAYER['Gatys et al. (2015)'],
                 style_layers: list=STYLE_LAYERS['Gatys et al. (2015)'],
                 is_notebook: bool=False) -> None:
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
        self.is_notebook = is_notebook

    @property
    def is_notebook(self) -> bool:
        """Return a bool determining if Jupyter features are enabled."""
        return self._is_notebook

    @is_notebook.setter
    def is_notebook(self, is_notebook: bool) -> None:
        self._is_notebook = is_notebook
        # set the tqdm pointer based on the setting
        if is_notebook:
            self._tqdm = tqdm_notebook
        else:
            self._tqdm = tqdm

    def __repr__(self) -> str:
        """Return a debugging representation of self."""
        return REPR.format(*[
            self.__class__.__name__,
            self.content_weight,
            self.style_weight,
            self.variation_weight,
            self.content_layer,
            self.style_layers,
            self.is_notebook
        ])

    def __call__(self, canvas: Canvas, iterations: int=1) -> Image:
        """
        Optimize the loss between the content and style of the given canvas.

        Args:
            canvas: the canvas of content and styles to synthesize from
            iterations: the number of iterations of optimization to run

        Returns: a synthesized image based on content and style of the canvas
        """
        # instantiate a new VGG_19 model. include_top tells the model to
        # not include the fully connected layers at the end (not interested
        # in classification). We use the input tensor from the canvas to define
        # the input dimension of the network. This allows us to synthesize
        # images of whatever size we want. Lastly, like Gatys et al. (2015)
        # we replace the MaxPooling layers with AveragePooling layers with
        # the `pooling` keyword argument
        model = VGG_19(input_tensor=canvas.input_tensor,
                       include_top=False,
                       pooling='avg')
        # create a dictionary of the layers in the model
        layers = {layer.name: layer.output for layer in model.layers}

        # the loss variable for accumulating the loss
        loss = backend.variable(0.0)

        # Calculate the content loss
        layer = layers[self.content_layer]
        # the content image is the first item in the layer tensor
        content = layer[0, :, :, :]
        # the output image is the last (3rd) item in the layer tensor
        combination = layer[2, :, :, :]
        # multiply the content loss by the content weight
        loss += self.content_weight * content_loss(content, combination)

        # iterate over the style layers to stylize the image
        for layer_name in self.style_layers:
            layer = layers[layer_name]
            # the style image is in the 2nd item in the layer tensor
            style = layer[1, :, :, :]
            # the output image is the last (3rd) item in the layer tensor
            combination = layer[2, :, :, :]
            # multiply the style weight (average over the layers) times the
            # style loss for the current layer
            sl = style_loss(style, combination, canvas.width, canvas.height)
            loss += (self.style_weight / len(self.style_layers)) * sl

        # apply the weighted variation loss for noise
        loss += self.variation_weight * total_variation_loss(canvas)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for step in range(iterations):
            sess.run([train_op, loss], feed_dict={canvas.output: x})
        # # define the gradients of the loss function and the canvas output
        # grads = backend.gradients(loss, canvas.output)

        # # define a function to add the gradients to the loss
        # outputs = [loss]
        # outputs += grads
        # f_outputs = backend.function([canvas.output], outputs)

        # def eval_loss_and_grads(x):
        #     x = x.reshape((1, canvas.height, canvas.width, 3))
        #     outs = f_outputs([x])
        #     loss_value = outs[0]
        #     grad_values = outs[1].flatten().astype('float64')
        #     return loss_value, grad_values

        # # build the evaluator and optimize the image
        # evaluator = Evaluator(eval_loss_and_grads)
        # image = self._optimize(evaluator, canvas, iterations)

        x = x.reshape((canvas.height, canvas.width, 3))
        x = denormalize(x)
        x = matrix_to_image(x)

        # clear the keras session to clear memory
        backend.clear_session()

        return x

    def _optimize(self, evaluator: Evaluator, canvas: Canvas, iterations: int):
        """
        """
        x = canvas.random_noise

        for i in self._tqdm(range(iterations)):
            x, min_val, info = optimize(evaluator.loss, x.flatten(),
                                        fprime=evaluator.grads, maxfun=20)

        # reshape, denormalize, and convert to an image object
        x = x.reshape((canvas.height, canvas.width, 3))
        x = denormalize(x)
        x = matrix_to_image(x)

        return x


# explicitly specify the public API of the module
__all__ = ['NeuralAlgorithmOfArtisticStyle']
