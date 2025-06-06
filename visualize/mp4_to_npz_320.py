import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path

# -------- 설정 --------
video_path = "./02148_H_A_BY_C4.mp4"
save_path = "./02148_H_A_BY_C4.npz"
model_path = "yolo11n-pose.pt"
target_frame_count = 50
interval_seconds = 0.2

# -------- 모델 불러오기 --------
yolo_model = YOLO(model_path)

# -------- 비디오 열기 --------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
interval_frame = int(fps * interval_seconds)
print(f"[INFO] FPS: {fps:.2f} | 추출 간격: {interval_frame}프레임마다")

pose_list = []
fallback_pose = None
frame_count = 0
log_info = []

# -------- Letterbox 함수 --------
def letterbox(img, target_size=(320, 320), color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_top = (target_size[0] - nh) // 2
    pad_bottom = target_size[0] - nh - pad_top
    pad_left = (target_size[1] - nw) // 2
    pad_right = target_size[1] - nw - pad_left
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)
    return padded, scale, pad_left, pad_top

# -------- 프레임 처리 --------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[INFO] 영상 끝 도달.")
        break

    if frame_count % interval_frame == 0:
        resized = cv2.resize(frame, (320, 240))
        letterboxed, scale, pad_left, pad_top = letterbox(resized)

        results = yolo_model(letterboxed)
        keypoints = results[0].keypoints
        info = {
            "frame": frame_count,
            "detected": False,
            "total_persons": 0,
            "valid_keypoints": 0,
            "fallback_used": False
        }

        if keypoints is not None and len(keypoints) > 0:
            info["detected"] = True
            info["total_persons"] = len(keypoints)

            valid_found = False
            for i in range(len(keypoints)):
                kpts = keypoints.xy[i].cpu().numpy()
                if kpts.shape == (17, 2):
                    # 원래 위치로 역보정 (optional), 여기서는 정규화만 함
                    kpts[:, 0] /= 320
                    kpts[:, 1] /= 320
                    if not valid_found:
                        pose_list.append(kpts)
                        fallback_pose = kpts
                        valid_found = True
                    info["valid_keypoints"] += 1

            if not valid_found and fallback_pose is not None:
                pose_list.append(fallback_pose)
                info["fallback_used"] = True
        else:
            if fallback_pose is not None:
                pose_list.append(fallback_pose)
                info["fallback_used"] = True

        log_info.append(info)

        if len(pose_list) >= target_frame_count:
            break

    frame_count += 1

cap.release()

# -------- 로그 출력 --------
print("\n[Detection Summary]")
for info in log_info:
    status = "✅" if info["detected"] else "❌"
    fb = " (fallback)" if info["fallback_used"] else ""
    print(f"Frame {info['frame']:4d}: {status} | "
          f"Persons: {info['total_persons']}, "
          f"Valid: {info['valid_keypoints']}{fb}")

# -------- 결과 저장 --------
if len(pose_list) < target_frame_count:
    print(f"[ERROR] 총 {len(pose_list)}개의 skeleton만 추출됨 (50개 필요)")
else:
    pose_array = np.stack(pose_list)                    # (50, 17, 2)
    label_array = np.zeros((target_frame_count,))       # dummy label
    np.savez(save_path, pose=pose_array, label=label_array)
    print(f"[✔] 저장 완료: {save_path} (pose: {pose_array.shape}, label: {label_array.shape})")
