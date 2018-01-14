# Download the VGG weights and configuration data

# download everything
all: download_layer_configuration download_weights

# the link to fetch the weights data
WEIGHTS=http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel

# download the weights data
download_weights:
	wget ${WEIGHTS}

# the link to fetch the layers file
LAYERS=https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f02f8769e64494bcd3d7e97d5d747ac275825721/VGG_ILSVRC_19_layers_deploy.prototxt

# download the layers file
download_layer_configuration:
	wget ${LAYERS}
