"A Neural Algorithm of Artistic Style."
from .vgg19 import VGG_19


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
