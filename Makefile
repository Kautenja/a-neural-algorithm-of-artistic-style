# the executable for Python
PYTHON=python3
# the build directory
BUILD=build
# the number of frames of interpolation to use in videos
INTER_FRAMES=3

all: style_transfer_videos layer_videos reconstruction_videos optimizer_videos tv_loss_videos phot_realistic_style_transfer_videos

# install Python dependencies in the requirements.txt
install:
	${PYTHON} -m pip install -r requirements.txt

# make a build directory
build:
	mkdir -p ${BUILD}

# Convert the frames in a given directory to a video of the directories name
# in its parent directory.
# Args:
#   1: the name of the directory in the build directory to find frames in
define frames_to_video
	${PYTHON} frames_to_video.py build/$(1) build/$(1).mp4 ${INTER_FRAMES}
endef

style_transfer_videos: build
	$(call frames_to_video,style-transfer/seated-nude)
	$(call frames_to_video,style-transfer/the-starry-night)
	$(call frames_to_video,style-transfer/the-scream)
	$(call frames_to_video,style-transfer/the-shipwreck-of-the-minotaur)
	$(call frames_to_video,style-transfer/composition-vii)
	$(call frames_to_video,style-transfer/houses-of-parliament)

content_layer_videos: build
	$(call frames_to_video,content-layer/block1_conv1)
	$(call frames_to_video,content-layer/block2_conv1)
	$(call frames_to_video,content-layer/block3_conv1)
	$(call frames_to_video,content-layer/block4_conv1)
	$(call frames_to_video,content-layer/block5_conv1)

style_layer_videos: build
	$(call frames_to_video,style-layer/block1_conv1)
	$(call frames_to_video,style-layer/block2_conv1)
	$(call frames_to_video,style-layer/block3_conv1)
	$(call frames_to_video,style-layer/block4_conv1)
	$(call frames_to_video,style-layer/block5_conv1)

layer_videos: content_layer_videos style_layer_videos

content_reconstruction_videos: build
	$(call frames_to_video,content-reconstruction/block1_conv1)
	$(call frames_to_video,content-reconstruction/block2_conv1)
	$(call frames_to_video,content-reconstruction/block3_conv1)
	$(call frames_to_video,content-reconstruction/block4_conv1)
	$(call frames_to_video,content-reconstruction/block4_conv2)
	$(call frames_to_video,content-reconstruction/block5_conv1)

style_reconstruction_videos: build
	$(call frames_to_video,style-reconstruction/block1_conv1)
	$(call frames_to_video,style-reconstruction/block2_conv1)
	$(call frames_to_video,style-reconstruction/block3_conv1)
	$(call frames_to_video,style-reconstruction/block4_conv1)
	$(call frames_to_video,style-reconstruction/block5_conv1)

reconstruction_videos: content_reconstruction_videos style_reconstruction_videos

optimizer_videos: build
	$(call frames_to_video,optimizers/Adam)
	$(call frames_to_video,optimizers/GradientDescent)
	$(call frames_to_video,optimizers/L_BFGS)

tv_loss_videos: build
	$(call frames_to_video,tv/0)
	$(call frames_to_video,tv/1)
	$(call frames_to_video,tv/10)
	$(call frames_to_video,tv/100)
	$(call frames_to_video,tv/1000)

phot_realistic_style_transfer_videos: build
	$(call frames_to_video,photo-realistic-style-transfer/Adam)
	$(call frames_to_video,photo-realistic-style-transfer/GradientDescent)
	$(call frames_to_video,photo-realistic-style-transfer/L_BFGS)
