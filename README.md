# A Neural Algorithm of Artistic Style (Keras Implementation)

An **implementation** of the arXiv preprint
[_A Neural Algorithm of Artistic Style [1]_](#references)
& paper
[_Image Style Transfer Using Convolutional Neural Networks [2]_](#references).

## Style Transfer

[style-transfer.ipynb](style-transfer.ipynb) describes the style transfer
process between a white noise image **x**, a content image **p**, and a style
representation **a**. Performing gradient descent of the content loss and
style loss with respect to **x** impressions the content of **p** into **x**,
bearing local styles, and colors from **a**.

<table>
    <tr>
        <td><b>Original Photograph</b> <i>Tubingen, Germany</i></td>
        <td></td>
        <td><img src="img/content/tubingen.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><b>Claude Monet</b> <i>Houses of Parliament</i></td>
        <td><img src="img/styles/houses-of-parliament.jpg" width="375"/></td>
        <td><img src="img/style-transfer/houses-of-parliament.jpg" width="425"/></td>
    </tr>
    <tr>
        <td><b>Pablo Picasso</b> <i>Seated Nude</i></td>
        <td><img src="img/styles/seated-nude.jpg" width="245"/></td>
        <td><img src="img/style-transfer/seated-nude.jpg" width="425"/></td>
    </tr>
    <tr>
        <td><b>Edvard Munch</b> <i>The Scream</i></td>
        <td><img src="img/styles/the-scream.jpg" width="250"/></td>
        <td><img src="img/style-transfer/the-scream.jpg" width="425"/></td>
    </tr>
    <tr>
        <td><b>Vincent van Gogh</b> <i>The Starry Night</i></td>
        <td><img src="img/styles/the-starry-night.jpg" width="400"/></td>
        <td><img src="img/style-transfer/the-starry-night.jpg" width="425"/></td>
    </tr>
    <tr>
        <td><b>William Turner</b> <i>The Shipwreck of The Minotaur</i></td>
        <td><img src="img/styles/the-shipwreck-of-the-minotaur.jpg" width="425"/></td>
        <td><img src="img/style-transfer/the-shipwreck-of-the-minotaur.jpg" width="400"/></td>
    </tr>
    <tr>
        <td><b>Wassily Kandinsky</b> <i>Composition VII</i></td>
        <td><img src="img/styles/composition-vii.jpg" width="425"/></td>
        <td><img src="img/style-transfer/composition-vii.jpg" width="425"/></td>
    </tr>
</table>

## Content Reconstruction

[content-reconstruction.ipynb](content-reconstruction.ipynb) describes the
content reconstruction process from white noise. Performing gradient descent
of the content loss on a white noise input **x** for a given content **p**
yields a representation of the networks activation for a given layer _l_.

<table>
    <tr>
        <th>Layer</th>
        <th>Result</th>
    </tr>
    <tr>
        <td><code>block1_conv1</code></td>
        <td><img src="img/content-reconstruction/block1_conv1.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>block2_conv1</code></td>
        <td><img src="img/content-reconstruction/block2_conv1.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>block3_conv1</code></td>
        <td><img src="img/content-reconstruction/block3_conv1.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>block4_conv1</code></td>
        <td><img src="img/content-reconstruction/block4_conv1.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>block4_conv2</code></td>
        <td><img src="img/content-reconstruction/block4_conv2.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>block5_conv1</code></td>
        <td><img src="img/content-reconstruction/block5_conv1.jpg" width="375"/></td>
    </tr>
</table>

## Style Reconstruction

[style-reconstruction.ipynb](style-reconstruction.ipynb) describes the style
reconstruction process on Wassily Kandinsky's <i>Composition VII</i> from white
noise. Performing gradient descent of the style loss on a white noise input
**x** for a given artwork **a** yields a representation of the networks
activation for a given set of layers _L_.

<table>
    <tr>
        <th>Layer</th>
        <th>Result</th>
    </tr>
    <tr>
        <td><code>block1_conv1</code></td>
        <td><img src="img/style-reconstruction/block1_conv1.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>block1_conv1</code>, <code>block2_conv1</code></td>
        <td><img src="img/style-reconstruction/block2_conv1.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>block1_conv1</code>, <code>block2_conv1</code>, <code>block3_conv1</code></td>
        <td><img src="img/style-reconstruction/block3_conv1.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>block1_conv1</code>, <code>block2_conv1</code>, <code>block3_conv1</code>, <code>block4_conv1</code></td>
        <td><img src="img/style-reconstruction/block4_conv1.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>block1_conv1</code>, <code>block2_conv1</code>, <code>block3_conv1</code>, <code>block4_conv1</code>, <code>block5_conv1</code></td>
        <td><img src="img/style-reconstruction/block5_conv1.jpg" width="375"/></td>
    </tr>
</table>

## Content Layer

[content-layer.ipynb](content-layer.ipynb) visualizes how the style transfer
is affected by using different layers for content loss.

<table>
    <tr>
        <th>Layer</th>
        <th>Result</th>
    </tr>
    <tr>
        <td><code>block1_conv1</code></td>
        <td><img src="img/content-layer/block1_conv1.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>block2_conv1</code></td>
        <td><img src="img/content-layer/block2_conv1.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>block3_conv1</code></td>
        <td><img src="img/content-layer/block3_conv1.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>block4_conv1</code></td>
        <td><img src="img/content-layer/block4_conv1.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>block5_conv1</code></td>
        <td><img src="img/content-layer/block5_conv1.jpg" width="375"/></td>
    </tr>
</table>

## Style Layers

[style-layers.ipynb](style-layers.ipynb) visualizes how the style transfer is
affected by using different sets of layers for style loss.

<table>
    <tr>
        <th>Layers</th>
        <th>Result</th>
    </tr>
    <tr>
        <td><code>block1_conv1</code></td>
        <td><img src="img/style-layer/block1_conv1.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>block1_conv1</code>, <code>block2_conv1</code></td>
        <td><img src="img/style-layer/block2_conv1.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>block1_conv1</code>, <code>block2_conv1</code>, <code>block3_conv1</code></td>
        <td><img src="img/style-layer/block3_conv1.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>block1_conv1</code>, <code>block2_conv1</code>, <code>block3_conv1</code>, <code>block4_conv1</code></td>
        <td><img src="img/style-layer/block4_conv1.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>block1_conv1</code>, <code>block2_conv1</code>, <code>block3_conv1</code>, <code>block4_conv1</code>, <code>block5_conv1</code></td>
        <td><img src="img/style-layer/block5_conv1.jpg" width="375"/></td>
    </tr>
</table>

## Optimizers

[optimizers.ipynb](optimizers.ipynb) employs _gradient descent_, _adam_, and
_L-BFGS_ to understand the effect of different black-box optimizers. Gatys et.
al use L-BFGS, but Adam appears to produce comparable results without as much
overhead.

<table>
    <tr>
        <th>Gradient Descent</th>
        <th>Adam</th>
        <th>L-BFGS</th>
    </tr>
    <tr>
        <td><img src="img/optimizers/GradientDescent.jpg" width="256"/></td>
        <td><img src="img/optimizers/Adam.jpg" width="256"/></td>
        <td><img src="img/optimizers/L_BFGS.jpg" width="256"/></td>
    </tr>
</table>

<p align="center">
<img src="img/optimizers/plot.png"/>
</p>

## TV Loss

[tv-loss.ipynb](tv-loss.ipynb) introduces total-variation loss to reduce
impulse noise in the images.

<table>
    <tr>
        <th>TV Loss Scale Factor</th>
        <th>Result</th>
    </tr>
    <tr>
        <td><code>0</code></td>
        <td><img src="img/tv/0.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>1</code></td>
        <td><img src="img/tv/1.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>10</code></td>
        <td><img src="img/tv/10.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>100</code></td>
        <td><img src="img/tv/100.jpg" width="375"/></td>
    </tr>
    <tr>
        <td><code>1000</code></td>
        <td><img src="img/tv/1000.jpg" width="375"/></td>
    </tr>
</table>

## Photo-Realistic Style Transfer

[photo-realistic-style-transfer.ipynb](photo-realistic-style-transfer.ipynb)
describes the photo-realistic style transfer process. Opposed to transferring
style from an artwork, this notebook explores transferring a nighttime style
from a picture of Piedmont Park at night to a daytime picture of Piedmont Park.

<table>
    <tr>
        <th>Content</th>
        <th>Style</th>
        <th>Result</th>
    </tr>
    <tr>
        <td><img src="img/photo-realistic-style-transfer/content.jpg" width="256"/></td>
        <td><img src="img/photo-realistic-style-transfer/style.jpg" width="256"/></td>
        <td><img src="img/photo-realistic-style-transfer/piedmont-park-L_BFGS.jpg" width="256"/></td>
    </tr>
</table>

## References

[_[1] L. A. Gatys, A. S. Ecker, and M. Bethge. A neural algorithm of artistic style. arXiv preprint
arXiv:1508.06576, 2015._][ref1]

[ref1]: https://arxiv.org/abs/1508.06576

[_[2] L. A. Gatys, A. S. Ecker, and M. Bethge. Image style transfer using convolutional neural networks. In
Computer Vision and Pattern Recognition (CVPR), 2016 IEEE Conference on, pages 2414–2423.
IEEE, 2016._][ref2]

[ref2]: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
