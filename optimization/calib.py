from openvino.tools.pot import DataLoader, IEEngine, compress_model_weights, load_model, save_model
from openvino.tools.pot.graph import load_model as pot_load_model
from openvino.tools.pot.engines import IEEngine
from openvino.tools.pot.algorithms.quantization.accuracy_aware import AccuracyAwareQuantization

model_config = {
    "model_name": "yolov8n_pose",
    "model": "yolov8n-pose.xml",
    "weights": "yolov8n-pose.bin"
}

engine_config = {
    "device": "CPU",
    "stat_requests_number": 1,
    "eval_requests_number": 1
}

algorithms = [{
    "name": "DefaultQuantization",  # or "AccuracyAwareQuantization"
    "params": {
        "preset": "performance",
        "stat_subset_size": 300
    }
}]

# 사용자 정의 DataLoader 구현 필요
class MyDataLoader(DataLoader):
    def __len__(self):
        return len(calibration_images)

    def __getitem__(self, idx):
        # Return np.ndarray, dict format
        return {"input": calibration_images[idx]}

model = pot_load_model(model_config)
engine = IEEngine(config=engine_config, data_loader=MyDataLoader())
pipeline = AccuracyAwareQuantization(engine=engine, algorithms=algorithms)

compressed_model = pipeline.run(model=model)
compress_model_weights(compressed_model)

save_model(model=compressed_model, save_path="yolov8n-pose-int8", model_name="yolov8n-pose-int8")
