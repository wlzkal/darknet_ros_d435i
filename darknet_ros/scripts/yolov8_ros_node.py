import rospy
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
import yaml

class YOLOv8Node:
    def __init__(self):
        # Use the correct path to your YAML file
        with open('/home/nick/lee/src/darknet_ros_d435i/darknet_ros/config/yolov8n.yaml', 'r') as f:
            self.config = yaml.safe_load(f)

        # Load the PyTorch model
        self.model = torch.load(self.config['yolo_model']['weight_file']['name'], map_location='cpu', weights_only=False)

        # ROS subscription and publishing
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.bounding_boxes_pub = rospy.Publisher('/darknet_ros/bounding_boxes', BoundingBoxes, queue_size=1)
        self.detection_image_pub = rospy.Publisher('/darknet_ros/detection_image', Image, queue_size=1)

    def image_callback(self, data):
        # Convert ROS image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

        # Perform object detection with YOLOv8
        results = self.model(cv_image)

        # Publish bounding boxes and detection image
        bounding_boxes_msg = self.convert_results_to_bounding_boxes(results)
        self.bounding_boxes_pub.publish(bounding_boxes_msg)

        detection_image_msg = self.bridge.cv2_to_imgmsg(results.render()[0], encoding="bgr8")
        self.detection_image_pub.publish(detection_image_msg)

    def convert_results_to_bounding_boxes(self, results):
        bounding_boxes_msg = BoundingBoxes()
        for *box, conf, cls in results.xyxy[0]:
            bounding_box = BoundingBox()
            bounding_box.xmin = int(box[0])
            bounding_box.ymin = int(box[1])
            bounding_box.xmax = int(box[2])
            bounding_box.ymax = int(box[3])
            bounding_box.probability = float(conf)
            bounding_box.Class = self.config['yolo_model']['detection_classes']['names'][int(cls)]
            bounding_boxes_msg.bounding_boxes.append(bounding_box)
        return bounding_boxes_msg

if __name__ == '__main__':
    rospy.init_node('yolov8_node')
    YOLOv8Node()
    rospy.spin()

