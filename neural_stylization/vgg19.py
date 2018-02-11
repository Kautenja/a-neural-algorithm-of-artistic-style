"""A Keras implementation of the VGG 19 CNN model.

This object oriented designed is based on the original code from the Keras
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
from keras import backend as K


# the definition for a tensor type as defined by Keras
Tensor = Union[Input, Layer]


def download_include_top() -> str:
    """Download the ImageNet Weights with FC Layers and return the path."""
    from keras.utils.data_utils import get_file
    return get_file(
        'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
        'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5',
        file_hash='cbe5617147190e668d6c5d5026f83318'
    )


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
                - 'avg': this is the optional value for VGG19. Gatys et
                    al. find that this produces smoother synthetic images.

        Returns:
            None

        """
        # setup the private instance variables for this object
        self._include_top = include_top
        self._input_tensor = input_tensor
        self._pooling = pooling

        # build the input layer
        input_block = self._build_input_block()
        # build the main layers
        x = self._build_main_blocks(input_block)
        # build the output layers
        x = self._build_output_block(x)

        # call the super initializer
        super().__init__(input_block, x)

        # load the weights
        self.load_imagenet_weights()

        # set outputs as a dictionary of layer names to output variables
        self.output_tensors = {layer.name: layer.output for layer in self.layers}

    def __getitem__(self, key: str) -> Tensor:
        """
        Return the output of the given layer.

        Args:
            key: the key of the layer to get the output of

        Returns:
            the output of the layer in question

        """
        return self.output_tensors[key]

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
        """Return a debugging representation of this object."""
        # combine the class name with the data and unwrap (*) for format
        return '{}(include_top={}, input_tensor={}, pooling={})'.format(*[
            self.__class__.__name__,
            self.include_top,
            self.input_tensor,
            self.pooling
        ])

    def _build_input_block(self) -> Tensor:
        """
        Build and return the input block for the network.

        Returns:
            a tensor representing the network up to the input blocks

        """
        # make sure the channel format is setup correctly (TensorFlow)
        if K.image_data_format() != 'channels_last':
            raise ValueError('image_data_format should be: "channels_last"')

        # if classification, the shape is predefined (assume RGB channels)
        if self.include_top:
            # the default size for ImageNet images (224 x 224 x RGB)
            input_shape = (224, 224, 3)
        # no input_shape provided, image synthesis or feature extraction
        else:
            # the default size for an size in-specific RGB image
            input_shape = (None, None, 3)

        # setup the input tensor
        if self.input_tensor is None:
            # no input tensor specified, build a new one with the given shape
            return Input(shape=input_shape)
        elif not K.is_keras_tensor(self.input_tensor):
            # tensor provided, but not a Keras tensor, convert to Keras tensor
            return Input(tensor=self.input_tensor, shape=input_shape)
        else:
            # already a Keras tensor / input layer. return as is
            return self.input_tensor

    def _build_main_blocks(self, x: Tensor) -> Tensor:
        """
        Build and return the main blocks of the network.

        Args:
            x: the input blocks of the network

        Returns:
            a tensor representing the network up to the main blocks

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

        Returns:
            a tensor representing the network up to the output blocks

        """
        # check if the model is using the fully connected layers and build if
        # it is using them.
        if self.include_top:
            x = Flatten(name='flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(4096, activation='relu', name='fc2')(x)
            # use 1000 units in the output (number of classes in ImageNet)
            x = Dense(1000, activation='softmax', name='predictions')(x)

        return x

    def load_imagenet_weights(self) -> None:
        """Load the pre-trained ImageNet weights for this VGG19 model."""
        # check if the model is using the Fully Connected layers
        if self.include_top:
            # download the fully connected layer weights
            weights_path = download_include_top()
        else:
            # load the CNN layer weights from disk
            from os.path import dirname
            weights_path = '{}/data/notop.h5'.format(dirname(__file__))
        # load the weights into self
        self.load_weights(weights_path)


# explicitly export the public API of the module
__all__ = ['VGG_19']
