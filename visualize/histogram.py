import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Colab 경로 설정
root_dir = Path("/content/drive/MyDrive/pose_tensor_npz")

# 프레임별 skeleton 인식 성공률 저장 리스트
success_counts = []

# 모든 npz 파일 검색
for npz_file in root_dir.rglob("*.npz"):
    print(f"Processing: {npz_file}")
    data = np.load(npz_file)
    pose_data = data['pose']  # (20, 17, 2) 형태

    # 각 프레임에서 skeleton이 정상적으로 인식된 keypoint 수 확인
    for frame in pose_data:
        valid_keypoints = np.sum(np.all(frame > 0, axis=-1))  # (0, 0) 제외
        success_counts.append(valid_keypoints)

# 히스토그램으로 성공률 시각화
plt.figure(figsize=(10, 6))
plt.hist(success_counts, bins=range(0, 18), edgecolor='black')
plt.title("Skeleton Recognition Success per Frame")
plt.xlabel("Number of Recognized Keypoints")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.6)

# 결과 저장
plt.savefig("/content/drive/MyDrive/skeleton_success_histogram.png")
plt.show()

print("Analysis complete. Histogram saved at /content/drive/MyDrive/skeleton_success_histogram.png")
