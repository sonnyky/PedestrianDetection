# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
import time
import pyrealsense2 as rs

def click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pos_str = str(x) + ", " + str(y)
        print("x, y : ", pos_str)

if __name__ == "__main__":
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', click)

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920 , 1080, rs.format.bgr8, 30)

    # define the upper and lower boundaries of the HSV pixel
    # intensities to be considered 'skin'
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    # Start streaming

    profile = pipeline.start(config)

    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Intrinsics & Extrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        color_image = cv2.resize(color_image, (0,0), fx=0.5, fy=0.5)
        # Visualization of the results of a detection.
        cv2.imshow("image", color_image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
