# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
import time
import pyrealsense2 as rs

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        #print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

if __name__ == "__main__":
    model_path = 'D:/Workspace/PedestrianDetection/pedestrian_detection/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # define the upper and lower boundaries of the HSV pixel
    # intensities to be considered 'skin'
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier("cascade.xml")

    # Start streaming

    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Intrinsics & Extrinsics
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(
            color_frame.profile)
        # Convert images to numpy arrays
        #depth_image = np.asanyarray(depth_frame.get_data())


        #####################################################################

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        ######################################################################

        img = color_image

        boxes, scores, classes, num = odapi.processFrame(img)

        # Visualization of the results of a detection.

        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)

                if(box[3] - box[1] > 0) and ( box[2] - box[0] > 0):
                    # Find contour of people inside the boxes (set ROI)
                    human_roi = img[box[0]:box[2], box[1]:box[3]]


                    #converted = cv2.cvtColor(human_roi, cv2.COLOR_BGR2HSV)
                    #skinMask = cv2.inRange(converted, lower, upper)

                    # apply a series of erosions and dilations to the mask
                    # using an elliptical kernel
                    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                    #skinMask = cv2.erode(skinMask, kernel, iterations=2)
                    #skinMask = cv2.dilate(skinMask, kernel, iterations=2)

                    # blur the mask to help remove noise, then apply the
                    # mask to the frame
                    #skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
                    #skin = cv2.bitwise_and(human_roi, human_roi, mask=skinMask)

                    #cv2.imshow("images", np.hstack([human_roi, skin]))


                    human_roi_gray = cv2.cvtColor(human_roi, cv2.COLOR_BGR2GRAY);
                    #ret, thresh = cv2.threshold(human_roi_gray, 127, 255, 0)

                    # Detect faces in the image
                    faces = faceCascade.detectMultiScale(
                        human_roi_gray,
                        scaleFactor=1.1,
                        minNeighbors=5
                    )
                    print ("Found {0} faces!".format(len(faces)))
                    thresh = cv2.Canny(human_roi_gray,180,200)
                    cv2.imshow("images", thresh)


                    # show the skin in the image along with the mask

                # Find contours in the roi
                    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, contours, -1, (128, 255, 255),
                    3, cv2.LINE_AA, hierarchy, 2, (box[1],box[0]) )


####################################################################################################################

                midpoint_x = np.int(box[1] + ((box[3] - box[1]) / 2))

                depth_top = aligned_depth_frame.get_distance(midpoint_x, box[0] + 20)
                depth_bottom = aligned_depth_frame.get_distance(midpoint_x, box[2])
                print("midpoint_x : " + str(midpoint_x))
                depth_top_point = rs.rs2_deproject_pixel_to_point(
                                color_intrin, [midpoint_x, box[0]+20], depth_top)

                depth_bottom_point = rs.rs2_deproject_pixel_to_point(
                    color_intrin, [midpoint_x, box[2]], depth_bottom)

####################################################################################################################



                human_height_vector = np.asanyarray(depth_bottom_point) - np.asanyarray(depth_top_point)

                human_height = np.linalg.norm(human_height_vector)
                human_height = np.round(human_height, 3)

                height_value = str(human_height) + " m"

                cv2.putText(img, height_value, (box[1], box[0]+35), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                print("*************************************************************************************")
                print("depth_top_point : ", depth_top_point)
                print("depth_bottom_point : ", depth_bottom_point)
                print("human height vector : ", human_height_vector)
                print("height of object : ", human_height);

        cv2.imshow("preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
