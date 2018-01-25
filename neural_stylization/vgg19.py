"""A keras implementation of the VGG 19 CNN model.

This object oriented designed is based on the original code from the keras
team here:
https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
"""
from typing import Union
from keras.models import Model
from keras.layers import Layer
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape


# the definition for a tensor type as defined by the keras documentation:
# https://keras.io/backend/
# scroll to the `is_keras_tensor` section for a description of this type
Tensor = Union[Input, Layer]


# the path to the pretrained weights
WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
# the name for the weights file on disk
WEIGHTS_FILE = 'vgg19_weights_tf_dim_ordering_tf_kernels.h5'
# the hash for the weights file
WEIGHTS_HASH = 'cbe5617147190e668d6c5d5026f83318'


# the path to the pretrained weights without the top (fully connected layers)
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
# the name for the weights (no top) file on disk
WEIGHTS_FILE_NO_TOP = 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
# the hash for the weights (no top) file
WEIGHT_HASH_NO_TOP = '253f8cb515780f3b799900260a226db6'


class VGG_19(Model):
    """The VGG 19 image recognition network."""

    def __init__(self,
                 include_top: bool=True,
                 input_tensor: Union[None, Input]=None,
                 pooling: str='max') -> None:
        """
        Initialize a new VGG19 network.

        Args:
            include_top: whether to include the 3 fully-connected
                layers at the top of the network.
                - True: includes the dense layers at the end for classification
                    on the 1000 ImageNet classes
                - False: leaves the fully connected layers out for feature
                    extraction or image synthesis or whatever other application
            input_tensor: optional Keras tensor (i.e. output of `Input()`)
                to use as image input for the model.
            pooling: the kind of pooling layers to use
                - 'max': this is the default value for standard VGG19. This
                    pooling method produces cleaner classification results
                - 'average': this is the optional value for VGG19. Gatys et
                    al. find that this produces smoother synthetic images.

        Returns: None
        """
        # setup the private instance variables for this object
        self._include_top = include_top
        self._input_tensor = input_tensor
        self._pooling = pooling

        # build the input layer
        img_input = self._build_input_block()
        # build the main layers
        x = self._build_main_blocks(img_input)
        # build the output layers
        x = self._build_output_block(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input

        # call the super initializer
        super().__init__(inputs, x, name=self.__class__.__name__)

        # load the weights
        self.load_weights()

    @property
    def include_top(self):
        """Return the immutable include_top flag for this network."""
        return self._include_top

    @property
    def input_tensor(self):
        """Return the immutable input tensor for the network."""
        return self._input_tensor

    @property
    def pooling(self):
        """Return the method used in the pooling layers of the network."""
        return self._pooling

    def __repr__(self):
        template = '{}(include_top={}, input_tensor={}, pooling={})'
        """Return a debugging representation of this object."""
        # combine the class name with the data and unwrap (*) for format
        return template.format(*[
            self.__class__.__name__,
            self.include_top,
            self.input_tensor,
            self.pooling
        ])

    def _build_input_block(self) -> Tensor:
        """
        Build and return the input block for the network

        Returns: a tensor representing the network up to the input blocks
        """
        # Determine proper input shape
        # can probably remove this. Check here for how:
        # https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
        input_shape = _obtain_input_shape(None,
                                          default_size=224,
                                          min_size=48,
                                          data_format=K.image_data_format(),
                                          require_flatten=self.include_top,
                                          weights='imagenet')
        # setup the input tensor
        if self.input_tensor is None:
            # no input tensor specified, build a blank one
            return Input(shape=input_shape)
        elif not K.is_keras_tensor(self.input_tensor):
            # tensor provided, but probably a numpy array, create a keras (tf)
            # tensor and return it
            return Input(tensor=self.input_tensor, shape=input_shape)
        else:
            # the tensor is already set up, return it as is
            return self.input_tensor

    def _build_main_blocks(self, x: Tensor) -> Tensor:
        """
        Build and return the main blocks of the network.

        Args:
            x: the input blocks of the network

        Returns: a tensor representing the network up to the main blocks
        """
        # setup the pooling layer initializer
        if self.pooling == 'avg':
            Pool2D = AveragePooling2D
        elif self.pooling == 'max':
            Pool2D = MaxPooling2D
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = Pool2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = Pool2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
        x = Pool2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
        x = Pool2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
        x = Pool2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        return x

    def _build_output_block(self, x: Tensor) -> Tensor:
        """
        Build and return the output block for the network.

        Args:
            x: the existing layers in the model to build onto

        Returns: a tensor representing the network up to the output blocks
        """
        # if include_top is set, build the fully connected output block
        if self.include_top:
            x = Flatten(name='flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(4096, activation='relu', name='fc2')(x)
            # use 1000 units in the output (number of classes in ImageNet)
            x = Dense(1000, activation='softmax', name='predictions')(x)

        return x

    def load_weights(self) -> None:
        """Load the weights for this VGG19 model."""
        # dox for the get_file method:
        # https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file
        # check if the top layers (fully connected) are included
        if self.include_top:
            # the path for weights WITH the top (fully connected) layers
            weights_path = get_file(WEIGHTS_FILE, WEIGHTS_PATH,
                                    file_hash=WEIGHTS_HASH)
        else:
            # the path for weights WITHOUT the top (fully connected) layers
            weights_path = get_file(WEIGHTS_FILE_NO_TOP, WEIGHTS_PATH_NO_TOP,
                                    file_hash=WEIGHT_HASH_NO_TOP)
        # load the weights into self
        super().load_weights(weights_path)


# explicitly export the public API of the module
__all__ = ['VGG_19']
