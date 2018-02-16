# the executable for Python
PYTHON=python3
# the directory the main framework exists in
MAIN=neural_stylization
# the directory with the dataset
DATASET=${MAIN}/dataset
# the build directory
BUILD=build
# the pointer to the pdflatex binary
PDFLATEX=pdflatex
# the pointer to the bibilography engine (bibtex)
BIBTEX=bibtex

# install Python dependencies in the requirements.txt
install:
	${PYTHON} -m pip install -r requirements.txt

# make a build directory
build:
	mkdir -p ${BUILD}

# make the presentation
presentation: build
	cd tex && pdflatex presentation
	cp tex/presentation.pdf ${BUILD}

# make the presentation using bibtex to compile references
presentation_w_ref: build
	cd tex && \
		${PDFLATEX} presentation && \
		${BIBTEX} presentation && \
		${PDFLATEX} presentation && \
		${PDFLATEX} presentation
	cp tex/presentation.pdf ${BUILD}

# make the review without references
review: build
	cd tex && ${PDFLATEX} review
	cp tex/review.pdf ${BUILD}

# make the review using bibtex to compile references
review_w_ref: build
	cd tex && \
		${PDFLATEX} review && \
		${BIBTEX} review && \
		${PDFLATEX} review && \
		${PDFLATEX} review
	cp tex/review.pdf ${BUILD}

# Convert the frames in a given directory to a video of the directories name
# in its parent directory.
# Args:
#   1: the name of the directory in the build directory to find frames in
define frames_to_video
	${PYTHON} frames_to_video.py build/$(1) build/$(1).mp4
endef

# make all the content reconstruction videos
content_videos:
	$(call frames_to_video,content/block1_conv1)
	$(call frames_to_video,content/block2_conv1)
	$(call frames_to_video,content/block3_conv1)
	$(call frames_to_video,content/block4_conv1)
	$(call frames_to_video,content/block4_conv2/sgd)
	$(call frames_to_video,content/block4_conv2/lbfgs)
	$(call frames_to_video,content/block5_conv1)

# make all the style reconstruction videos
style_videos:
	$(call frames_to_video,style/block1_conv1)
	$(call frames_to_video,style/block2_conv1)
	$(call frames_to_video,style/block3_conv1)
	$(call frames_to_video,style/block4_conv1)
	$(call frames_to_video,style/block5_conv1)
	$(call frames_to_video,style/starry_night/sgd)
	$(call frames_to_video,style/starry_night/lbfgs)

# make all the style transfer videos
transfer_videos:
	$(call frames_to_video,transfer/kandinsky/sgd)
	$(call frames_to_video,transfer/kandinsky/lbfgs)
	$(call frames_to_video,transfer/monet/sgd)
	$(call frames_to_video,transfer/monet/lbfgs)
	$(call frames_to_video,transfer/scream/sgd)
	$(call frames_to_video,transfer/scream/lbfgs)
	$(call frames_to_video,transfer/seated-nudes/sgd)
	$(call frames_to_video,transfer/seated-nudes/lbfgs)
	$(call frames_to_video,transfer/shipwreck/sgd)
	$(call frames_to_video,transfer/shipwreck/lbfgs)
	$(call frames_to_video,transfer/starry-night/sgd)
	$(call frames_to_video,transfer/starry-night/lbfgs)

# make all the built videos
videos: content_videos style_videos transfer_videos
