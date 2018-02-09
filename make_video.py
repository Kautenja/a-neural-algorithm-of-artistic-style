"""This file turns a directory of png files into video file."""
import cv2
import re
from sys import argv
from os import listdir


# get the directory from the command line input
directory = argv[1]
video_name = argv[2]

# get the frames from the given directory in sorted order
frames = [filename for filename in listdir(directory) if '.png' in filename]
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
for frame in frames:
	out.write(frame)

# cleanup and release the video
out.release()
cv2.destroyAllWindows()
