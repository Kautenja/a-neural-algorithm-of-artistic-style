# A Neural Algorithm of Artistic Style (Implementation)

An **implementation** of the arXiv preprint
[_A Neural Algorithm of Artistic Style [1]_](#references)
& paper
[_Image Style Transfer Using Convolutional Neural Networks [2]_](#references).

## Original Photograph: _Tubingen, Germany_

<p float="left" align="center">
<img src="img/content/tubingen.jpg" width="375"/>
</p>

-----

## Claude Monet's _Houses of Parliament_

<table>
<tr>
<td>
<img src="img/styles/houses-of-parliament.jpg" width="375"/>
</td>
<td>
<img src="img/transfer/houses-of-parliament-tv-1e0.png" width="425"/>
</td>
</tr>
</table>

## Pablo Picasso's _Seated Nude_

<table>
<tr>
<td>
<img src="img/styles/seated-nude.jpg" width="245"/>
</td>
<td>
<img src="img/transfer/seated-nude.png" width="425"/>
</td>
</tr>
</table>

## Edvard Munch's _The Scream_

<table>
<tr>
<td>
<img src="img/styles/the-scream.jpg" width="250"/>
</td>
<td>
<img src="img/transfer/the-scream.png" width="425"/>
</td>
</tr>
</table>

## Vincent van Gogh's _The Starry Night_

<table>
<tr>
<td>
<img src="img/styles/the-starry-night.jpg" width="400"/>
</td>
<td>
<img src="img/transfer/the-starry-night.png" width="425"/>
</td>
</tr>
</table>

## William Turner's _The Shipwreck of The Minotaur_

<table>
<tr>
<td>
<img src="img/styles/the-shipwreck-of-the-minotaur.jpg" width="425"/>
</td>
<td>
<img src="img/transfer/the-shipwreck-of-the-minotaur.png" width="400"/>
</td>
</tr>
</table>

## Wassily Kandinsky's _Composition VII_

<table>
<tr>
<td>
<img src="img/styles/composition-vii.jpg" width="425"/>
</td>
<td>
<img src="img/transfer/composition-vii.png" width="425"/>
</td>
</tr>
</table>

# Getting Started

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes. See deployment for notes
on how to deploy the project on a live system.

## Prerequisites

1.  [python3][] - Programming Environment
1.  [pip3][] - Python Dependency Management

[python3]: https://python.org
[pip3]: https://packaging.python.org/tutorials/installing-packages/

## Installing

To install dependent modules:

```shell
pip3 install -r requirements.txt
```

# Project Components

[neural_stylization][] contains Python modules with utility methods and
classes for the project.

[neural_stylization]: neural_stylization

## VGG19

This project relies on the VGG19 architecture.
[VGG19-classification.ipynb][] outlines some basic image classification
using the network with weight-set **W** pre-trained on the ImageNet
dataset. The implementation of VGG19 can be found in
[neural_stylization/vgg19.py][]. Utility methods for loading manipulating,
and normalizing image can be found in [neural_stylization/img_util.py][].

[VGG19-classification.ipynb]: VGG19-classification.ipynb
[neural_stylization/vgg19.py]: neural_stylization/vgg19.py
[neural_stylization/img_util.py]: neural_stylization/img_util.py

## Content Reconstruction

[content-reconstruction.ipynb][] describes the content reconstruction
process from white noise. Performing gradient descent of the content loss
on a white noise input **x** for a given content **p** yields a
representation of the networks activation for a given layer _l_.

[content-reconstruction.ipynb]: content-reconstruction.ipynb

## Style Reconstruction

[style-reconstruction.ipynb][] describes the style reconstruction
process from white noise. Performing gradient descent of the style loss
on a white noise input **x** for a given artwork **a** yields a
representation of the networks activation for a given set of layers _L_.

[style-reconstruction.ipynb]: style-reconstruction.ipynb

## Style Transfer

[style-transfer.ipynb][] describes the style transfer process between a white
noise image **x**, a content image **p**, and a style representation **a**.
Performing gradient descent of the content loss and style loss with respect
to **x** impressions the content of **p** into **x**, bearing local styles,
and colors from **a**.

[style-transfer.ipynb]: style-transfer.ipynb

## Photo-Realistic Style Transfer

[photo-realistic-style-transfer.ipynb][] describes the photo-realistic style
transfer process. Opposed to transfering style from an artwork, this notebook
explores transfering a nighttime theme from a picture of one city to a
daytime picture of another city with mixed results.

[photo-realistic-style-transfer.ipynb]: photo-realistic-style-transfer.ipynb

## Content Layer Selection

[effect-of-content-layer.ipynb][] visualizes how the style transfer is affected
by using different layers for content loss.

[effect-of-content-layer.ipynb]: effect-of-content-layer.ipynb

## Style Layer Selection

[effect-of-style-layers.ipynb][] visualizes how the style transfer is affected
by using different sets of layers for style loss.

[effect-of-style-layers.ipynb]: effect-of-style-layers.ipynb

## Optimizers

[optimizers.ipynb][] employs _gradient descent_, _adam_, and _L-BFGS_ to
understand the affect of different blackbox optimizers. Gatys et. al use
L-BFGS, but Adam appears to produce competetive results too.

[optimizers.ipynb]: optimizers.ipynb

# Acknowledgments

-   [keras-team](https://github.com/keras-team) provides `Keras`, a high
    level neural network framework. They also provide the pre-trained
    ImageNet weights and some tutorials that help build this project.

# References

[_[1] L. A. Gatys, A. S. Ecker, and M. Bethge. A neural algorithm of artistic style. arXiv preprint
arXiv:1508.06576, 2015._][ref1]

[ref1]: https://arxiv.org/abs/1508.06576

[_[2] L. A. Gatys, A. S. Ecker, and M. Bethge. Image style transfer using convolutional neural networks. In
Computer Vision and Pattern Recognition (CVPR), 2016 IEEE Conference on, pages 2414–2423.
IEEE, 2016._][ref2]

[ref2]: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
