# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

# Person detection with tensorflow and estimate height by using trigonometry on the bounding box.
# need to have knowledge of camera height and pitch

# A simple webcam is used

import numpy as np
import tensorflow as tf
import cv2
import time
import sys
sys.path.insert(0,'D:/Workspace/PedestrianDetection/pedestrian_detection/utils')
from measurement import measurement

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        measure_instance = measurement()
        measure_instance.set_image_positions(refPt[0][0], refPt[0][1])
        measure_instance.set_camera_parameters(43.603, 43.603)
        measure_instance.set_camera_pitch_and_height(60, 840)

        horizontal_angle = measure_instance.calc_horizontal_angle()
        vertical_angle = measure_instance.calc_vertical_angle()

        #print("horizontal angle : " + str(horizontal_angle))
        #print("vertical angle : " + str(vertical_angle))

        position_top = measure_instance.calc_3d_position(vertical_angle, horizontal_angle)
        debug_string = "3d position of point : " + str(position_top[0]) + ", " + str(
            position_top[1]) + ", " + str(position_top[2])
        print(debug_string)

        #object_height = measure_instance.calc_height_object_on_floor(position_floor[1], position_top[1])
        #print("Estimated human height is : " + str(object_height))
        #print(debug_string)

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
    threshold = 0.9
    trackerType = "MEDIANFLOW"

    print('Starting application ...')
    msr = measurement()
    msr.test_call_method(10, 15)

    ## Counter data holder
    people_count = 0

    ## Select boxes
    bboxes = []
    colors = []
    persons = []

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click)

    cap = cv2.VideoCapture(1)

    while True:

        r, img = cap.read()
        boxes, scores, classes, num = odapi.processFrame(img)

        # Visualization of the results of a detection.
        num_of_people = 0
        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                num_of_people += 1
                box = boxes[i]
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                human_roi = (box[1],box[0], box[3] - box[1],box[2] - box[1])
                if len(bboxes) < num_of_people and box[3] < 620 and box[1] > 50:
                    print("there are untracked people")
                    bboxes.append(human_roi)
                    one_tracker = createTrackerByName(trackerType)
                    one_tracker.init(img, human_roi)
                    persons.append(one_tracker)
                    #print("current bboxes length : ", len(bboxes))
                    colors.append((np.random.randint(64, 255), np.random.randint(64, 255), np.random.randint(64, 255)))
                    #multiTracker.add(createTrackerByName(trackerType), img, human_roi)

                # get distance of person from the camera, from the middle pixel
                midpoint_x = np.int(box[1] + ((box[3] - box[1]) / 2))
                midpoint_y = np.int(box[0] + ((box[2] - box[0]) / 2))

        for j in range(len(persons)):
            if j < len(persons):
                ok, bbox = persons[j].update(img)
                if ok:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(img, p1, p2, colors[j], 2, 1)

                    # Calculate detected person's height here which is estimated from the bounding box height
                    msr.set_camera_parameters(43.603, 43.603)
                    msr.set_camera_pitch_and_height(60, 2340)

                    msr.set_image_positions(bbox[0]+(bbox[2]/2), bbox[1])

                    cv2.circle(img, (int(bbox[0] + (bbox[2]/2)), int(bbox[1])), 5, (0,255,0), -1)
                    horizontal_angle = msr.calc_horizontal_angle()
                    vertical_angle = msr.calc_vertical_angle()
                    position_top = msr.calc_3d_position(vertical_angle, horizontal_angle)[:]


                    msr.set_image_positions(bbox[0] + (bbox[2]/2), bbox[1] + bbox[3])
                    cv2.circle(img, (int(bbox[0] + (bbox[2] / 2)), int(bbox[1]+bbox[3])), 5, (0, 255, 0), -1)

                    horizontal_angle = msr.calc_horizontal_angle()
                    vertical_angle = msr.calc_vertical_angle()

                    position_floor = msr.calc_3d_position(vertical_angle, horizontal_angle)

                    human_height = msr.calc_height_object_on_floor(position_floor[1], position_top[1])
                    people_height_string = str(human_height) + " m"
                    cv2.putText(img, people_height_string, (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


                    if bbox[0] < 50 or bbox[0] + bbox[2] > 620:
                        people_count += 1
                        persons.pop(j)
                        num_of_people -= 1
                        bboxes.pop(j)
                else:
                    # Tracking failure
                    cv2.putText(img, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    persons.pop(j)
                    num_of_people -= 1
                    bboxes.pop(j)
        #print("Number of people detected : ", num_of_people)
        people_counter_string = str(people_count) + " people"
        cv2.putText(img, people_counter_string, (50,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 255), 2)

        cv2.imshow("image", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
