# A Neural Algorithm of Artistic Style (Keras Implementation)

An **implementation** of the arXiv preprint
[_A Neural Algorithm of Artistic Style [1]_](#references)
& paper
[_Image Style Transfer Using Convolutional Neural Networks [2]_](#references).

## Getting Started

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes. See deployment for notes
on how to deploy the project on a live system.

### Prerequisites

1.  [python3][] - Programming Environment
1.  [pip3][] - Python Dependency Management

[python3]: https://python.org
[pip3]: https://packaging.python.org/tutorials/installing-packages/

### Installing

To install dependent modules:

```shell
pip3 install -r requirements.txt
```

## Usage

[src][] contains Python modules with utility methods and
classes for the project.

[src]: src

<!-- ### VGG19 -->

<!-- This project relies on the VGG19 architecture.
[VGG19-classification.ipynb][] outlines some basic image classification
using the network with weight-set **W** pre-trained on the ImageNet
dataset. The implementation of VGG19 can be found in
[src/vgg19.py][]. Utility methods for loading manipulating,
and normalizing image can be found in [src/img_util.py][]. -->

<!-- [VGG19-classification.ipynb]: VGG19-classification.ipynb
[src/vgg19.py]: src/vgg19.py
[src/img_util.py]: src/img_util.py -->

### Content Reconstruction

[content-reconstruction.ipynb][] describes the content reconstruction
process from white noise. Performing gradient descent of the content loss
on a white noise input **x** for a given content **p** yields a
representation of the networks activation for a given layer _l_.

[content-reconstruction.ipynb]: content-reconstruction.ipynb

<table>
    <tr>
        <th>Layer</th>
        <th>Result</th>
    </tr>
    <tr>
        <td><code>block1_conv1</code></td>
        <td><img src="img/content-reconstruction/block1_conv1.png" width="375"/></td>
    </tr>
    <tr>
        <td><code>block2_conv1</code></td>
        <td><img src="img/content-reconstruction/block2_conv1.png" width="375"/></td>
    </tr>
    <tr>
        <td><code>block3_conv1</code></td>
        <td><img src="img/content-reconstruction/block3_conv1.png" width="375"/></td>
    </tr>
    <tr>
        <td><code>block4_conv1</code></td>
        <td><img src="img/content-reconstruction/block4_conv1.png" width="375"/></td>
    </tr>
    <tr>
        <td><code>block4_conv2</code></td>
        <td><img src="img/content-reconstruction/block4_conv2.png" width="375"/></td>
    </tr>
    <tr>
        <td><code>block5_conv1</code></td>
        <td><img src="img/content-reconstruction/block5_conv1.png" width="375"/></td>
    </tr>
</table>

### Style Reconstruction

[style-reconstruction.ipynb][] describes the style reconstruction
process from white noise. Performing gradient descent of the style loss
on a white noise input **x** for a given artwork **a** yields a
representation of the networks activation for a given set of layers _L_.

[style-reconstruction.ipynb]: style-reconstruction.ipynb

<table>
    <tr>
        <th>Layer</th>
        <th>Result</th>
    </tr>
    <tr>
        <td><code>block1_conv1</code></td>
        <td><img src="img/style-reconstruction/block1_conv1.png"/></td>
    </tr>
    <tr>
        <td><code>block2_conv1</code></td>
        <td><img src="img/style-reconstruction/block2_conv1.png"/></td>
    </tr>
    <tr>
        <td><code>block3_conv1</code></td>
        <td><img src="img/style-reconstruction/block3_conv1.png"/></td>
    </tr>
    <tr>
        <td><code>block4_conv1</code></td>
        <td><img src="img/style-reconstruction/block4_conv1.png"/></td>
    </tr>
    <tr>
        <td><code>block5_conv1</code></td>
        <td><img src="img/style-reconstruction/block5_conv1.png"/></td>
    </tr>
</table>

### Style Transfer

[style-transfer.ipynb][] describes the style transfer process between a white
noise image **x**, a content image **p**, and a style representation **a**.
Performing gradient descent of the content loss and style loss with respect
to **x** impressions the content of **p** into **x**, bearing local styles,
and colors from **a**.

[style-transfer.ipynb]: style-transfer.ipynb

<table>
    <tr>
        <td><b>Original Photograph</b> <i>Tubingen, Germany</i></td>
        <td></td>
        <td><img src="img/content/tubingen.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><b>Claude Monet</b> <i>Houses of Parliament</i></td>
        <td><img src="img/styles/houses-of-parliament.jpg" width="375"/></td>
        <td><img src="img/transfer/houses-of-parliament.png" width="425"/></td>
    </tr>
    <tr>
        <td><b>Pablo Picasso</b> <i>Seated Nude</i></td>
        <td><img src="img/styles/seated-nude.jpg" width="245"/></td>
        <td><img src="img/transfer/seated-nude.png" width="425"/></td>
    </tr>
    <tr>
        <td><b>Edvard Munch</b> <i>The Scream</i></td>
        <td><img src="img/styles/the-scream.jpg" width="250"/></td>
        <td><img src="img/transfer/the-scream.png" width="425"/></td>
    </tr>
    <tr>
        <td><b>Vincent van Gogh</b> <i>The Starry Night</i></td>
        <td><img src="img/styles/the-starry-night.jpg" width="400"/></td>
        <td><img src="img/transfer/the-starry-night.png" width="425"/></td>
    </tr>
    <tr>
        <td><b>William Turner</b> <i>The Shipwreck of The Minotaur</i></td>
        <td><img src="img/styles/the-shipwreck-of-the-minotaur.jpg" width="425"/></td>
        <td><img src="img/transfer/the-shipwreck-of-the-minotaur.png" width="400"/></td>
    </tr>
    <tr>
        <td><b>Wassily Kandinsky</b> <i>Composition VII</i></td>
        <td><img src="img/styles/composition-vii.jpg" width="425"/></td>
        <td><img src="img/transfer/composition-vii.png" width="425"/></td>
    </tr>
</table>

### Photo-Realistic Style Transfer

[photo-realistic-style-transfer.ipynb][] describes the photo-realistic style
transfer process. Opposed to transfering style from an artwork, this notebook
explores transfering a nighttime theme from a picture of one city to a
daytime picture of another city with mixed results.

[photo-realistic-style-transfer.ipynb]: photo-realistic-style-transfer.ipynb

### Content Layer Selection

[effect-of-content-layer.ipynb][] visualizes how the style transfer is affected
by using different layers for content loss.

[effect-of-content-layer.ipynb]: effect-of-content-layer.ipynb

<table>
    <tr>
        <th>Layer</th>
        <th>Result</th>
    </tr>
    <tr>
        <td><code>block1_conv1</code></td>
        <td><img src="img/layer-content/block1_conv1.png"/></td>
    </tr>
    <tr>
        <td><code>block2_conv1</code></td>
        <td><img src="img/layer-content/block2_conv1.png"/></td>
    </tr>
    <tr>
        <td><code>block3_conv1</code></td>
        <td><img src="img/layer-content/block3_conv1.png"/></td>
    </tr>
    <tr>
        <td><code>block4_conv1</code></td>
        <td><img src="img/layer-content/block4_conv1.png"/></td>
    </tr>
    <tr>
        <td><code>block5_conv1</code></td>
        <td><img src="img/layer-content/block5_conv1.png"/></td>
    </tr>
</table>

### Style Layer Selection

[effect-of-style-layers.ipynb][] visualizes how the style transfer is affected
by using different sets of layers for style loss.

[effect-of-style-layers.ipynb]: effect-of-style-layers.ipynb

<table>
    <tr>
        <th>Layers</th>
        <th>Result</th>
    </tr>
    <tr>
        <td><code>block1_conv1</code></td>
        <td><img src="img/layer-style/block1_conv1.png"/></td>
    </tr>
    <tr>
        <td><code>block1_conv1</code>, <code>block2_conv1</code></td>
        <td><img src="img/layer-style/block2_conv1.png"/></td>
    </tr>
    <tr>
        <td><code>block1_conv1</code>, <code>block2_conv1</code>, <code>block3_conv1</code></td>
        <td><img src="img/layer-style/block3_conv1.png"/></td>
    </tr>
    <tr>
        <td><code>block1_conv1</code>, <code>block2_conv1</code>, <code>block3_conv1</code>, <code>block4_conv1</code></td>
        <td><img src="img/layer-style/block4_conv1.png"/></td>
    </tr>
    <tr>
        <td><code>block1_conv1</code>, <code>block2_conv1</code>, <code>block3_conv1</code>, <code>block4_conv1</code>, <code>block5_conv1</code></td>
        <td><img src="img/layer-style/block5_conv1.png"/></td>
    </tr>
</table>

### Optimizers

[optimizers.ipynb][] employs _gradient descent_, _adam_, and _L-BFGS_ to
understand the affect of different blackbox optimizers. Gatys et. al use
L-BFGS, but Adam appears to produce competetive results too.

[optimizers.ipynb]: optimizers.ipynb

## References

[_[1] L. A. Gatys, A. S. Ecker, and M. Bethge. A neural algorithm of artistic style. arXiv preprint
arXiv:1508.06576, 2015._][ref1]

[ref1]: https://arxiv.org/abs/1508.06576

[_[2] L. A. Gatys, A. S. Ecker, and M. Bethge. Image style transfer using convolutional neural networks. In
Computer Vision and Pattern Recognition (CVPR), 2016 IEEE Conference on, pages 2414–2423.
IEEE, 2016._][ref2]

[ref2]: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
