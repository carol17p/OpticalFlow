# OpticalFlow

This calculates dense optical flow using the Farneback method implemented in OpenCV for all mp4 videos in a specified folder.
The file blurred_flow.py also uses a GaussianBlur in an attempt to reduce noise in the video and produce better results.

A folder to store the resulting output video files should be created.

## Usage:
$ python flow.py --input [path to input video files] --output [path to output video files]
