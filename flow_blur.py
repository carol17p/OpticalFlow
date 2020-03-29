# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:56:48 2020

@author: carol
"""

import cv2
import numpy as np
import glob
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video files")
ap.add_argument("-o", "--output", required=True,
	help="path to output video files")
args = vars(ap.parse_args())
d=1

for filename in glob.glob(args["input"]+ "\*.mp4"):
    vc = cv2.VideoCapture(filename)

    ret, first_frame = vc.read()
    first_frame = cv2.resize(first_frame, (800, 600), interpolation = cv2.INTER_AREA)
    # prev_gray = cv2.GaussianBlur(first_frame, (11, 11), 0)
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(first_frame)
    # Sets image saturation to maximum
    mask[..., 1] = 255
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    name = args["output"] + "\\result_blur_%d.mp4"%d
    out = cv2.VideoWriter(name,fourcc,25,(first_frame.shape[1], first_frame.shape[0]))
    d+=1
    
    while(vc.isOpened()):
    
        ret, frame = vc.read()
        if(ret!=True):
            break
        frame = cv2.resize(frame, (800, 600), interpolation = cv2.INTER_AREA)

    
        # gray = cv2.GaussianBlur(frame, (11, 11), 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # gray = cv2.resize(gray, None, fx=scale, fy=scale)
    
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.2, flags = 0)
        # Compute the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Set image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2
        # Set image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        
        #frame = cv2.resize(frame, None, fx=scale, fy=scale)
    
        dense_flow = cv2.addWeighted(frame, 1,rgb, 1, 0)
        # cv2.imshow("Dense optical flow", dense_flow)
        out.write(dense_flow)
        
        prev_gray = gray
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    vc.release()
    out.release()
    cv2.destroyAllWindows()
