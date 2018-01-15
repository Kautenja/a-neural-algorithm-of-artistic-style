"A Neural Algorithm of Artistic Style."
import numpy as np
from .vgg19 import VGG_19


# the mean pixel values from the paper
MEAN_PIXEL = np.array([123.68 ,  116.779,  103.939])


class NeuralAlgorithmOfArtisticStyle(VGG_19):
	"""A Neural Algorithm of Artistic Style."""

	def __init__(self):
		"""Initialize a new neural algorithm of artistic style."""
		# call super with the parameters from the paper.
		# include_top = False  # this turns off the 3 fully connected layers
		#                      # at the end of the network
		# pooling = 'avg'      # turns on average pooling on the output layer
		#                      # TODO: the paper mentions "replacing the max-
		#                      # pooling operation by average pooling improves
		#                      # gradient flow and one receives more appealing
		#                      # results". A) is this true? B) does this apply
		#                      # to this layer at all, or instead the
		#                      # MaxPooling layers in the network?
		super().__init__(include_top=False, pooling='avg')


# explicitly specify the public API of the module
__all__ = ['NeuralAlgorithmOfArtisticStyle']
