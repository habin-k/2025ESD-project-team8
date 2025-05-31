import cv2
import os
import numpy as np

# ====== 설정 ======
video_path = "00015_H_A_SY_C4.mp4"     # 4K 영상 경로
output_dir = "calib_images"          # 저장할 폴더
target_size = (640, 640)             # YOLO 입력 사이즈
interval_sec = 0.5                   # 몇 초마다 한 장 저장할지

# ====== 폴더 생성 ======
os.makedirs(output_dir, exist_ok=True)

# ====== 영상 로드 ======
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * interval_sec)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"영상 FPS: {fps}, 총 프레임 수: {frame_count}, 저장 간격: {frame_interval}프레임마다")

def letterbox(img, target_size=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    target_w, target_h = target_size

    # scale 비율 유지하면서 resize
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 패딩 적용
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded

saved = 0
for idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break

    if idx % frame_interval == 0:
        resized = letterbox(frame, target_size)
        filename = os.path.join(output_dir, f"frame_{saved:03d}.jpg")
        cv2.imwrite(filename, resized)
        print(f"✅ Saved: {filename}")
        saved += 1

cap.release()
print(f"📁 총 {saved}장의 이미지가 저장되었습니다.")