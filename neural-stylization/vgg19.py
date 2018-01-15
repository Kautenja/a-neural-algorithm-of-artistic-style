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

# the format template for the representation of VGG_19
REPR = '{}(include_top={}, weights={}, input_tensor={}, input_shape={}, pooling={}, classes={})'


class VGG_19(Model):
    """The VGG 19 image recognition architecture."""

    def __init__(self,
                 include_top: bool=True,
                 weights: Union[None, str]='imagenet',
                 input_tensor: Union[None, Input]=None,
                 input_shape: Union[None, tuple]=None,
                 pooling: Union[None, str]=None,
                 classes: int=1000) -> None:
        """
        Instantiates the VGG19 architecture.

        Optionally loads weights pre-trained
        on ImageNet. Note that when using TensorFlow,
        for best performance you should set
        `image_data_format='channels_last'` in your Keras config
        at ~/.keras/keras.json.
        The model and the weights are compatible with both
        TensorFlow and Theano. The data format
        convention used by the model is the one
        specified in your Keras config file.

        Args:
            include_top: whether to include the 3 fully-connected
                layers at the top of the network.
            weights: one of `None` (random initialization),
                  'imagenet' (pre-training on ImageNet),
                  or the path to the weights file to be loaded.
            input_tensor: optional Keras tensor (i.e. output of `Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `channels_last` data format)
                or `(3, 224, 224)` (with `channels_first` data format).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 48.
                E.g. `(200, 200, 3)` would be one valid value.
            pooling: Optional pooling mode for feature extraction
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
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.

        Returns: None
        """
        if not (weights in {'imagenet', None} or os.path.exists(weights)):
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization), `imagenet` '
                             '(pre-training on ImageNet), '
                             'or the path to the weights file to be loaded.')

        if weights == 'imagenet' and include_top and classes != 1000:
            raise ValueError('If using `weights` as imagenet with `include_top`'
                             ' as true, `classes` should be 1000')

        # store the variables for use by __repr__ and __str__
        self.init_args = [
        	include_top,
        	weights,
        	input_tensor,
        	input_shape,
        	pooling,
        	classes
        ]

        # build the input layer
        img_input = self._build_input_layer(include_top,
        									weights,
        									input_tensor,
        									input_shape)
        # build the main layers
        x = self._build_main_layers(img_input)
        # build the output layers
        x = self._build_output_layers(x, include_top, pooling, classes)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input

        # call the super initializer
        super(VGG_19, self).__init__(inputs, x, name=self.__class__.__name__)

        # load the weights
        self._load_weights(include_top, weights)

    def __repr__(self):
        """Return a debugging representation of this object."""
        # combine the class name with the data and unwrap (*) for format
        return REPR.format(*[self.__class__.__name__] + self.init_args)

    def _build_input_layer(self,
                            include_top: bool,
                            weights: Union[None, str],
                            input_tensor: Union[None, Input],
                            input_shape: Union[None, tuple]):
        # Determine proper input shape
        input_shape = _obtain_input_shape(input_shape,
                                          default_size=224,
                                          min_size=48,
                                          data_format=K.image_data_format(),
                                          require_flatten=include_top,
                                          weights=weights)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        return img_input

    def _build_main_layers(self, x: 'InputLayerTensor'):
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        return x

    def _build_output_layers(self,
                              x: 'MainLayersTensor',
                              include_top: bool,
                              pooling: Union[None, str],
                              classes: int):
        if include_top:
            # Classification block
            x = Flatten(name='flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(4096, activation='relu', name='fc2')(x)
            x = Dense(classes, activation='softmax', name='predictions')(x)
        else:
            if pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                x = GlobalMaxPooling2D()(x)

        return x

    def _load_weights(self, include_top: bool, weights: Union[None, str]):
        if weights == 'imagenet':
            if include_top:
                weights_path = get_file(WEIGHTS_FILE,
                                        WEIGHTS_PATH,
                                        cache_subdir='models',
                                        file_hash=WEIGHTS_HASH)
            else:
                weights_path = get_file(WEIGHTS_FILE_NO_TOP,
                                        WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        file_hash=WEIGHT_HASH_NO_TOP)
            self.load_weights(weights_path)
            if K.backend() == 'theano':
                layer_utils.convert_all_kernels_in_model(self)

            if K.image_data_format() == 'channels_first':
                if include_top:
                    maxpool = self.get_layer(name='block5_pool')
                    shape = maxpool.output_shape[1:]
                    dense = self.get_layer(name='fc1')
                    layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

                if K.backend() == 'tensorflow':
                    warnings.warn('You are using the TensorFlow backend, yet '
                                  'you are using the Theano '
                                  'image data format convention '
                                  '(`image_data_format="channels_first"`). '
                                  'For best performance, set '
                                  '`image_data_format="channels_last"` in '
                                  'your Keras config '
                                  'at ~/.keras/keras.json.')
        elif weights is not None:
            self.load_weights(weights)


# explicitly export the public API of the module
__all__ = ['VGG_19']
