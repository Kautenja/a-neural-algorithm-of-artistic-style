"""A keras implementation of the VGG 19 CNN model.

This object oriented designed is based on the original code from the keras
team here:
https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
"""
from typing import Union
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape


# the name of the weights dataset
WEIGHTS = 'imagenet'
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

# the number of classes in the dataset
CLASSES = 1000

# the format template for the representation of VGG_19
REPR = '{}(include_top={}, input_tensor={}, pooling={}, global_pooling={})'


class VGG_19(Model):
    """The VGG 19 image recognition network."""

    def __init__(self,
                 include_top: bool=True,
                 input_tensor: Union[None, Input]=None,
                 pooling: str='max',
                 global_pooling: Union[None, str]=None) -> None:
        """
        Initialize a new VGG19 network.

        Args:
            include_top: whether to include the 3 fully-connected
                layers at the top of the network.
            input_tensor: optional Keras tensor (i.e. output of `Input()`)
                to use as image input for the model.
            global_pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional layer.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional layer, and thus
                    the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                    be applied.

        Returns: None
        """
        # store the variables for use by __repr__ and __str__
        self.init_args = [
            repr(include_top),
            repr(input_tensor),
            repr(pooling),
            repr(global_pooling)
        ]

        # build the input layer
        img_input = self._build_input_block(include_top, input_tensor)
        # build the main layers
        x = self._build_main_blocks(img_input, pooling)
        # build the output layers
        x = self._build_output_block(x, include_top, global_pooling)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input

        # call the super initializer
        super(VGG_19, self).__init__(inputs, x, name=self.__class__.__name__)

        # load the weights
        self._load_weights(include_top)

    def __repr__(self):
        """Return a debugging representation of this object."""
        # combine the class name with the data and unwrap (*) for format
        return REPR.format(*[self.__class__.__name__] + self.init_args)

    def _build_input_block(self,
                           include_top: bool,
                           input_tensor: Union[None, Input]) -> 'tensor':
        """
        Build and return the input block for the network

        Args:
            include_top: whether to include the fully connected layers
            input_tensor: the input tensor if any was specified

        Returns: a tensor representing the network up to the input blocks
        """
        # Determine proper input shape
        input_shape = _obtain_input_shape(None,
                                          default_size=224,
                                          min_size=48,
                                          data_format=K.image_data_format(),
                                          require_flatten=include_top,
                                          weights=WEIGHTS)
        # return the appropriate input tensor
        if input_tensor is None:
            return Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                return Input(tensor=input_tensor, shape=input_shape)
            else:
                return input_tensor

    def _build_main_blocks(self, x: 'tensor', pooling: str) -> 'tensor':
        """
        Build and return the main blocks of the network.

        Args:
            x: the input blocks of the network
            pooling: the kind of pooling to use at the end of each block

        Returns: a tensor representing the network up to the main blocks
        """
        # setup the pooling layer initializer
        if pooling == 'avg':
            pool2d = AveragePooling2D
        elif pooling == 'max':
            pool2d = MaxPooling2D
        else:
            raise ValueError('`pooling` should be either: "avg", "max"')
        # build the blocks
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = pool2d((2, 2), strides=(2, 2), name='block1_pool')(x)
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = pool2d((2, 2), strides=(2, 2), name='block2_pool')(x)
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
        x = pool2d((2, 2), strides=(2, 2), name='block3_pool')(x)
        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
        x = pool2d((2, 2), strides=(2, 2), name='block4_pool')(x)
        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
        x = pool2d((2, 2), strides=(2, 2), name='block5_pool')(x)

        return x

    def _build_output_block(self,
                            x: 'tensor',
                            include_top: bool,
                            global_pooling: Union[None, str]) -> 'tensor':
        """
        Build and return the output block for the network.

        Args:
            x: the existing layers in the model to build onto
            include_top: whether to use the fully connected layers or the
                global pooling layers
            global_pooling: if `include_top` is False, the type of pooling to
                use. Either 'avg' or 'max'.

        Returns: a tensor representing the network up to the output blocks
        """
        # if include_top is set, build the fully connected output block
        if include_top:
            x = Flatten(name='flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(4096, activation='relu', name='fc2')(x)
            x = Dense(CLASSES, activation='softmax', name='predictions')(x)
        # otherwise if pooling is 'avg' return the global avg pooling block
        elif global_pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        # otherwise if pooling is 'avg' return the global max pooling block
        elif global_pooling == 'max':
            x = GlobalMaxPooling2D()(x)

        return x

    def _load_weights(self, include_top: bool) -> None:
        """
        Load the weights for this VGG19 model.

        Args:
            include_top: whether to include the fully connected layers

        Returns: None
        """
        # dox for the get_file method:
        # https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file
        # check if the top layers (fully connected) are included
        if include_top:
            # the path for weights WITH the top (fully connected) layers
            weights_path = get_file(WEIGHTS_FILE, WEIGHTS_PATH,
                                    file_hash=WEIGHTS_HASH)
        else:
            # the path for weights WITHOUT the top (fully connected) layers
            weights_path = get_file(WEIGHTS_FILE_NO_TOP, WEIGHTS_PATH_NO_TOP,
                                    file_hash=WEIGHT_HASH_NO_TOP)
        # load the weights into self
        self.load_weights(weights_path)


# explicitly export the public API of the module
__all__ = ['VGG_19']
