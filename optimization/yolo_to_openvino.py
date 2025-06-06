from ultralytics import YOLO

# 모델 로드
model = YOLO("yolo11n-pose.pt")

model.export(
    format="openvino",
    imgsz=512,        # 입력 해상도 설정 (정사각형이면 하나만 넣어도 됨)
    opset=12,
    dynamic=True,
    simplify=True
)