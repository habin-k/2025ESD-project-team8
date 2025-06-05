#from ultralytics import YOLO
#YOLO("yolo11n-pose.pt").export(format="openvino")

from ultralytics import YOLO
YOLO("yolov8n-pose.pt").export(format="openvino", imgsz=320)
