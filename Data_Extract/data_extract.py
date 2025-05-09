from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import torch

# YOLOv8 pose 모델 로드 (CUDA 적용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("yolov8n-pose.pt").to(device)

# 입력 경로 (source: mp4, labeling: json)
video_root = Path("/content/drive/MyDrive/Dataset_simplified/source")
labeling_root = Path("/content/drive/MyDrive/Dataset_simplified/labeling")
output_root = Path("/content/pose_tensor_npz")

# 영상 파일 순회
for video_path in tqdm(video_root.rglob("*.mp4")):
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    # JSON 파일 경로 추론
    json_path = labeling_root / video_path.relative_to(video_root).with_suffix(".json")
    if not json_path.exists():
        print(f"[❗JSON 파일 없음] {json_path}")
        continue

    # JSON 파일 로드
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        fall_start = json_data["sensordata"].get("fall_start_frame", 0)
        fall_end = json_data["sensordata"].get("fall_end_frame", 0)

    labels = []  # 각 프레임의 낙상 여부 (0 또는 1)

    # 20프레임 (0.5초 간격)
    for i in range(20):
        frame_idx = i * 30
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"[❗프레임 읽기 실패] {video_path.name} 프레임 {frame_idx}")
            pose = np.zeros((17, 2))
        else:
            try:
                # 프레임을 CUDA로 변환
                frame_gpu = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)

                # YOLOv8 모델 추론 (GPU에서 실행)
                results = model.predict(source=frame_gpu, save=False, verbose=False)
                keypoints = results[0].keypoints

                if keypoints is None or not hasattr(keypoints, "xyn") or len(keypoints.xyn) == 0:
                    print(f"[❗감지 실패] {video_path.name} 프레임 {frame_idx}")
                    pose = np.zeros((17, 2))
                else:
                    pose = keypoints.xyn[0].cpu().numpy()
                    if pose.shape != (17, 2):
                        print(f"[❗잘못된 shape] {video_path.name} 프레임 {frame_idx}, shape={pose.shape}")
                        pose = np.zeros((17, 2))
                    else:
                        print(f"[✅감지 성공] {video_path.name} 프레임 {frame_idx}")
            except Exception as e:
                print(f"[❗예외 발생] {video_path.name} 프레임 {frame_idx}: {e}")
                pose = np.zeros((17, 2))

        # 낙상 여부 라벨 (0 또는 1)
        is_fall = 1 if fall_start <= frame_idx <= fall_end else 0
        labels.append(is_fall)
        frames.append(pose)

    cap.release()

    # numpy 텐서 스택 (20, 17, 2)
    try:
        frames_np = np.stack(frames, axis=0)  # (20, 17, 2)
        labels_np = np.array(labels)          # (20,)
    except ValueError as e:
        print(f"[❌np.stack 실패] {video_path.name}, 오류: {e}")
        continue

    # 저장 경로 생성 및 저장 (.npz)
    save_path = output_root / video_path.relative_to(video_root).with_suffix(".npz")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, pose=frames_np, label=labels_np)
    print(f"[✅저장 완료] {save_path}")