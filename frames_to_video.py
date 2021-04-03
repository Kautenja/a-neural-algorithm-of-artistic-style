"""This file turns a directory of png files into video file."""
import cv2
import re
from sys import argv
from os import listdir


def pairs(iterable):
    """
    Return a new iterable over sequential pairs in the given iterable.
    i.e. (0,1), (1,2), ..., (n-2,n-1)
    Args:
        iterable: the iterable to iterate over the pairs of
    Returns: a new iterator over the pairs of the given iterator
    """
    # lazily import tee from `itertools`
    from itertools import tee
    # split the iterator into 2 identical iterators
    a, b = tee(iterable)
    # retrieve the next item from the b iterator
    next(b, None)
    # zip up the iterators of current and next items
    return zip(a, b)


# the directory of png files to make into a video
directory = argv[1]
# the name of the video file to write
video_name = argv[2]
# the number of frames to use for interpolation
interpolation_frames = int(argv[3])

# get the frames from the given directory in sorted order
frames = [filename for filename in listdir(directory) if '.jpg' in filename]
# use the integer filename value to account for the lack of 0 padding
frames = sorted(frames, key=lambda x: int(re.sub(r'[^0-9]+', "", x)))
# open the frames as images
frames = [cv2.imread('{}/{}'.format(directory, frame)) for frame in frames]
# unpack the dimensions of the images
height, width, dims = frames[0].shape

# setup a video output stream
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(video_name, fourcc, 20.0, (width, height))

# write the frames to the video file
for frame, next_frame in pairs(frames):
    # write the current frame
    out.write(frame)
    # iterate over the number of interpolation frames
    for i in range(1, interpolation_frames + 1):
        # calculate a frame between current and next frames
        w = i / interpolation_frames
        mid_frame = cv2.addWeighted(frame, 1 - w, next_frame, w, 0)
        # write this mid frame to the video
        out.write(mid_frame)

# the last frame still needs written
out.write(frames[-1])

# cleanup and release the video
out.release()
cv2.destroyAllWindows()
