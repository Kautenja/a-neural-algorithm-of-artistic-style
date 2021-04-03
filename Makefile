# the executable for Python
PYTHON=python3
# the build directory
BUILD=build
# the number of frames of interpolation to use in videos
INTER_FRAMES=3

all:
	@echo "dont use the generic 'make' in this project"

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

# make all the content reconstruction videos
content_videos: build
	$(call frames_to_video,content/block1_conv1)
	$(call frames_to_video,content/block2_conv1)
	$(call frames_to_video,content/block3_conv1)
	$(call frames_to_video,content/block4_conv1)
	$(call frames_to_video,content/block5_conv1)
	$(call frames_to_video,content/block4_conv2)

# make all the style reconstruction videos
style_videos: build
	$(call frames_to_video,style/block1_conv1)
	$(call frames_to_video,style/block2_conv1)
	$(call frames_to_video,style/block3_conv1)
	$(call frames_to_video,style/block4_conv1)
	$(call frames_to_video,style/block5_conv1)

# make all the style transfer videos
transfer_videos: build
	$(call frames_to_video,transfer/seated-nude)
	$(call frames_to_video,transfer/the-starry-night)
	$(call frames_to_video,transfer/the-scream)
	$(call frames_to_video,transfer/the-shipwreck-of-the-minotaur)
	$(call frames_to_video,transfer/composition-vii)
	$(call frames_to_video,transfer/houses-of-parliament/tv-0)
	$(call frames_to_video,transfer/houses-of-parliament/tv-1e0)
	$(call frames_to_video,transfer/houses-of-parliament/tv-1e1)

# make all the built videos
videos: content_videos style_videos transfer_videos
