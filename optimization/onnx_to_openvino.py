from openvino.tools.mo import convert_model

onnx_model_path = "yolov8n-pose.onnx"
ov_model = convert_model(
    onnx_model_path,
    input_shape=[1, 3, 640, 640],
    compress_to_fp16=False,  # True로 하면 FP16
)
from openvino.runtime import serialize
serialize(ov_model, "yolov8n-pose.xml", "yolov8n-pose.bin")
