from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")
model.export(format="onnx", opset=12, dynamic=True, simplify=True)
