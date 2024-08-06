import rospy
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
import yaml
import numpy as np
import cv2
from ultralytics import YOLO

class YOLOv8Node:
    def __init__(self):
        # Load configuration from YAML file
        with open('/home/nick/lee/src/darknet_ros_d435i/darknet_ros/config/yolov8n.yaml', 'r') as f:
            self.config = yaml.safe_load(f)

        # Load the PyTorch model
        self.load_model()

        # ROS subscription and publishing
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.bounding_boxes_pub = rospy.Publisher('/darknet_ros/bounding_boxes', BoundingBoxes, queue_size=1)
        self.detection_image_pub = rospy.Publisher('/darknet_ros/detection_image', Image, queue_size=1)

    def load_model(self):
        # Load the model using the YOLO class from Ultralytics
        model_path = self.config['yolo_model']['weight_file']['name']
        self.model = YOLO(model_path)  # Correctly instantiate the model

    def image_callback(self, data):
        # Convert ROS image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

        # Preprocess image for model input
        img = self.preprocess_image(cv_image)

        # Perform object detection with YOLOv8
        with torch.no_grad():
            results = self.model(img)

        # Debugging - print results
        print("Detection results:", results)

        # Proceed if there are results
        if results is None or len(results) == 0:
            rospy.logwarn("No objects detected.")
            return

        # Parse the results and publish bounding boxes and detection image
        bounding_boxes_msg = self.convert_results_to_bounding_boxes(results, cv_image.shape)
        self.bounding_boxes_pub.publish(bounding_boxes_msg)

        # Draw bounding boxes on the image
        self.draw_bounding_boxes(cv_image, bounding_boxes_msg)

        # Publish the image with bounding boxes
        detection_image_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        self.detection_image_pub.publish(detection_image_msg)

    def preprocess_image(self, image):
        # Resize and normalize the image
        img_size = self.config['model_parameters']['imgsz']
        img = cv2.resize(image, (img_size, img_size))
        img = img.astype(np.float32) / 255.0  # Normalize to 0-1
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, 0)  # Add batch dimension
        img = torch.from_numpy(img).to(self.config['model_parameters']['device'])
        return img

    def convert_results_to_bounding_boxes(self, results, image_shape):
        bounding_boxes_msg = BoundingBoxes()
        height, width = image_shape[:2]

        # Access the results directly as a list of bounding box objects
        if results and results[0].boxes is not None:
            for det in results[0].boxes.data:  # Ensure accessing the data correctly
                # det.xyxy is likely a torch tensor of bounding box coordinates
                x1, y1, x2, y2 = det[:4].cpu().numpy()  # Convert to numpy array
                conf = det[4].cpu().numpy()  # confidence
                cls = int(det[5].cpu().numpy())  # class index

                bounding_box = BoundingBox()
                bounding_box.xmin = int(x1)
                bounding_box.ymin = int(y1)
                bounding_box.xmax = int(x2)
                bounding_box.ymax = int(y2)
                bounding_box.probability = float(conf)
                bounding_box.Class = self.config['yolo_model']['detection_classes']['names'][cls]
                bounding_boxes_msg.bounding_boxes.append(bounding_box)

        return bounding_boxes_msg

    def draw_bounding_boxes(self, image, bounding_boxes_msg):
        # Draw the bounding boxes on the image
        for box in bounding_boxes_msg.bounding_boxes:
            # Draw rectangle on the image
            cv2.rectangle(image, (box.xmin, box.ymin), (box.xmax, box.ymax), (0, 255, 0), 2)

            # Put the class label and confidence score
            label = f"{box.Class}: {box.probability:.2f}"
            cv2.putText(image, label, (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == '__main__':
    rospy.init_node('yolov8_node')
    YOLOv8Node()
    rospy.spin()

