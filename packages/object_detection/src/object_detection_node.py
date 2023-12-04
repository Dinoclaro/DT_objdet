#!/usr/bin/env python3
import cv2
import math
import numpy as np
import rospy
import time
from typing import  Tuple

from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import  EpisodeStart, WheelsCmdStamped
from sensor_msgs.msg import CompressedImage

from nn_model.model import Wrapper
from nn_model.constants import IMAGE_SIZE, AREA, SCORE
from nn_model.constants import \
    NUMBER_FRAMES_SKIPPED, \
    filter_by_classes, \
    filter_by_bboxes, \
    filter_by_scores

class ObjectDetectionNode(DTROS):
    def __init__(self, node_name):

        # intialise the DTROS parent class
        super(ObjectDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.initialised = False
        self.log("Initialising...")

        # Get the vehicle name
        self.veh = rospy.get_namespace().strip("/")

        # BRAITENBERG parameters 
        self._time_debug = rospy.get_param("~time_debug", False) 
            # Controller constants
        self.gain: float = 1.0
        self.const: float = 0.3
        self.straight: float = 0.3
        self.stop: float = 0.0
        self.pwm_left = self.const
        self.pwm_right = self.const
            # Normalisation constants
        self.l_max = -math.inf
        self.r_max = -math.inf
        self.l_min = math.inf
        self.r_min = math.inf
            # Weight matrices 
        self.left  = None
        self.right = None
        
        # PUBLISHERS
        # TODO: investigate different queue sizes
            # Wheel commands
        wheels_cmd_topic = f"/{self.veh}/wheels_driver_node/wheels_cmd"
        self.pub_wheel_cmd = rospy.Publisher(
        wheels_cmd_topic, WheelsCmdStamped, 
        queue_size=1, 
        dt_topic_type=TopicType.CONTROL
        )

            # Detection image  
        self.pub_detections_img = rospy.Publisher(
            "~image/compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.DEBUG
        ) 

            # Debug image
        self.pub_wheel_cmd_img = rospy.Publisher(
            "~debug/debug_image/compressed",
            CompressedImage, 
            queue_size=1,
            dt_topic_type=TopicType.DEBUG
        )

        # SUBSCRIBERS
            # Episode start
            # TODO: investigate what this episode topic actually does
        episode_start_topic = f"/{self.veh}/episode_start"
        rospy.Subscriber(
            episode_start_topic,
            EpisodeStart,
            self.episode_start_cb,
            queue_size=1
        )

            # Camera image
        self.sub_image = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.image_cb,
            buff_size = 10000000,
            queue_size = 1
        )

        # BRIDGE
        self.bridge = CvBridge()

        # YOLO MODEL
            # Setting debug to True will display the image with the bounding boxes
        self._debug = rospy.get_param("~debug", False) 
            # Load Yolo model
        self.log("Loading model...")
        self.model_wrapper = Wrapper()
        self.log("Model loaded")
            # Initialise variables
        self.frame_id = 0
        self.debug_color = (0, 0, 255)
        self.first_image_received = False
        self. first_processing_done = False 

        # Execution times
        self.execution_times = []

        # Completed initialisation
        self.initialised = True
        self.log("Initialised")

    def episode_start_cb(self, msg: EpisodeStart):
        self.log("Episode started")
        self.pub_wheel_commands(self.stop, self.stop, msg.header)

    def image_cb(self, image_msg):
        '''
        Callback function for processing incoming image messages. Applies YOLOv5 object detection to the image, calls the compute_commands function if a valid detection and publishes the resulting image with bounding boxes and class labels.

        Args:
            image_msg (sensor_msgs.msg.Image): The incoming image message.
        '''
        # Do not move if NOT intialised
        if not self.initialised:
            self.pub_wheel_commands(self.stop, self.stop, image_msg.header)
            return
        
        # Check to call YOLO model 
        call = self.call_yolo()
        if not call:
            # If frame ID is not a factor use previous compute commands
            self.pub_wheel_commands(self.pwm_left, self.pwm_right, image_msg.header)
            return
        else:
            # Convert image message to cv2 image
            try: 
                bgr = self.bridge.compressed_imgmsg_to_cv2(image_msg)
            except ValueError as e:
                self.logger("Could not decode image: %s" % e)
                return
            
            # Convert to RGB
            rgb = bgr[..., ::-1]
            # resize for YOLO
            rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))

            # Run YOLO and only time if debug is True
            if self._time_debug:
                start_time = time.time()
                bboxes, classes, scores = self.model_wrapper.predict(rgb)
                self.log(f"YOLO bboxes: {bboxes}, classes: {classes}, scores: {scores}")

                self.log(f"YOLO runtime: {time.time() - start_time}")
                execution_time = time.time() - start_time
                self.execution_times.append(execution_time)
                min_time_yolo = min(self.execution_times)
                max_time_yolo = max(self.execution_times)
                avg_time_yolo = sum(self.execution_times) / len(self.execution_times)
                self.log(f"YOLO time: Min: {min_time_yolo} \n Max:{max_time_yolo} \n Avg:{avg_time_yolo}")
            else: 
                bboxes, classes, scores = self.model_wrapper.predict(rgb)
            
            # Filter detections
            detection, box, cla, sco = self.filterDet(bboxes, classes, scores)

            # Create detection map
            detection_map = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)

            if detection: 
                # populate detection map
                detection_map = self.drawBbox(detection_map, box)
                map = detection_map[:, :, 2]  # Index 0 corresponds to the red channel
                self.compute_commands(map)
        
                self.pub_wheel_commands(self.pwm_left, self.pwm_right, image_msg.header)
            else: 
                # Drive straight if there are no valid detections 
                self.pwm_left = self.straight
                self.pwm_right = self.straight
                self.pub_wheel_commands(self.pwm_left, self.pwm_right, image_msg.header) 

            # Publish image
            map_bgr = detection_map[..., ::-1]
            weight_img = self.bridge.cv2_to_compressed_imgmsg(map_bgr)
            self.pub_detections_img.publish(weight_img)

            # Publish debug image
            if self._debug:
                if box == None:
                   bgr = rgb[..., ::-1]
                else:
                   bgr = self.drawBbox(rgb, box)[..., ::-1]

                # Publish detection debug image
                obj_det_img = self.bridge.cv2_to_compressed_imgmsg(bgr)
                self.pub_wheel_cmd_img.publish(obj_det_img)

    def pub_wheel_commands(self, pwm_left, pwm_right, header):
        '''
        Publishes the wheel commands to the wheels_driver_node.
        '''
        wheel_control_msg = WheelsCmdStamped()
        wheel_control_msg.header = header

        # Wheel topic commands 
        wheel_control_msg.vel_left = pwm_left
        wheel_control_msg.vel_right = pwm_right

        if self._cmd_debug:
            self.log(f"Publishing wheel commands: L = {wheel_control_msg.vel_left}, R = {wheel_control_msg.vel_right}")
        # Publish wheel commands
        self.pub_wheel_cmd.publish(wheel_control_msg)

    def call_yolo(self):
        '''
        Returns True is the frame_id is a factor of NUMBER_FRAMES_SKIPPED, otherwise returns False.
        '''
        self.frame_id += 1
        self.frame_id = self.frame_id % (1 + NUMBER_FRAMES_SKIPPED())

        if self.frame_id != 0:
            return False
        else: 
            return True
        
    def filterDet(self, bboxes, classes, scores):
        '''
        Filters the detections by class, bounding box and score.
        '''
        filtered_data = list(filter(lambda x: filter_by_bboxes(x[0]) and filter_by_classes(x[1]) and filter_by_scores(x[2]), zip(bboxes, classes, scores, )))
        if len(filtered_data) > 0:
            return True, list(map(lambda x: x[0], filtered_data)), list(map(lambda x: x[1], filtered_data)), list(map(lambda x: x[2], filtered_data))
        else: 
            return False, None, None, None
        
    def drawBbox(self, rgb, bboxes):
        ''' 
        Draws bounding boxes given an image. 
        '''
        color = (0, 0, 255)

        for box in bboxes:
            pt1 = np.array([int(box[0]), int(box[1])])
            pt2 = np.array([int(box[2]), int(box[3])])

            pt1 = tuple(pt1)
            pt2 = tuple(pt2)
            rgb = cv2.rectangle(rgb, pt1, pt2, color, thickness = 2)
    
        return rgb
    
    def compute_commands(self, map):
        '''
        Computes the left and right PWM commands based on the input map.
        '''
        if map is None: 
            return 0.0
        
        if self.left is None: 
            # if it is the first time, we initialize the structures
            shape = map.shape[0], map.shape[1]
            self.left = self.get_motor_left_matrix(shape)
            self.right = self.get_motor_right_matrix(shape)
        
        # Compute the left and right activations 
        l = float(np.sum(map * self.left))
        r = float(np.sum(map * self.right))

        # Normalise the activations
        l_norm, r_norm = self.normalise(l, r)

        # Convert the activations to PWM commands
        self.pwm_to_cmd(l_norm, r_norm)

        self.log(f"ls: {self.pwm_left}, rs: {self.pwm_right}")

    def get_motor_left_matrix(self, shape: Tuple[int, int]) -> np.ndarray:
        '''
        Returns a matrix that represents the left half image activation based on the `weight` attribute.
        '''
        res = np.zeros(shape=shape, dtype="float32")
        res[:, :int(shape[1]/2)] = 1
        
        return res

    def get_motor_right_matrix(self, shape: Tuple[int, int]) -> np.ndarray:
        '''
        Returns a matrix that represents the right half image activation based on the `weight` attribute.
        '''
        res = np.zeros(shape=shape, dtype="float32")
        res[:, int(shape[1]/2):] = 1

        return res
    
    def rescale(self, a: float, L: float, U: float):
        '''
            Rescales a value to the range [0, 1] given the lower and upper bounds.
        '''
        if np.allclose(L, U):
            return 0.0
        elif a > U:
            return 1.0
        
        return (a - L) / (U - L)
    
    def normalise(self, l: float , r: float) -> Tuple[float, float]:
        '''
        Normalises the left and right values to the range [0, 1].
        '''
        self.l_max = max(self.l_max, l)
        self.r_max = max(self.r_max, r)
        self.l_min = min(self.l_min, l)
        self.r_min = min(self.r_min, r)

        l = self.rescale(l, self.l_min, self.l_max)
        r = self.rescale(r, self.r_min, self.r_max)

        return l, r

    def pwm_to_cmd(self, ls: float, rs: float): 
        '''
        Converts a PWM value to a command value.
        '''
        self.pwm_left = self.const + ls * self.gain
        # Max the pwm can be is (1*gain + const)
        self.pwm_left = self.rescale(self.pwm_left, 0, (self.gain + self.const))     
        
        self.pwm_right = self.const + rs * self.gain
        self.pwm_right = self.rescale(self.pwm_right, 0, (self.gain + self.const))  

    def pub_wheel_commands(self, pwm_left, pwm_right, header):
        '''
        Publishes the left and right PWM commands to the `pub_wheel_cmd` ROS topic.

        Args:
            pwm_left (float): The left PWM command.
            pwm_right (float): The right PWM command.
            header (std_msgs.msg.Header): The header for the `WheelsCmdStamped` message.

        Returns:
            None
        '''
        wheel_control_msg = WheelsCmdStamped()
        wheel_control_msg.header = header

        # Wheel topic commands
        wheel_control_msg.vel_left = pwm_left
        #self.log(f"vel_left: {wheel_control_msg.vel_left}")

        wheel_control_msg.vel_right = pwm_right
        #self.log(f"vel_right: {wheel_control_msg.vel_right}")

        self.pub_wheel_cmd.publish(wheel_control_msg)
    
if __name__ == "__main__":
    # Initialize the node
    object_detection_node = ObjectDetectionNode(node_name="object_detection_node")
    # Keep it spinning
    rospy.spin()