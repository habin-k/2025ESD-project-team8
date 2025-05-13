import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.notebook import tqdm

# Colab 경로 설정
root_dir = Path("/content/drive/MyDrive/pose_tensor_npz")

# 영상별 프레임 인식 성공률 저장 리스트
recognition_counts = []

# 모든 npz 파일 검색
for npz_file in tqdm(root_dir.rglob("*.npz"), desc="Processing Files"):
    print(f"Processing: {npz_file}")
    data = np.load(npz_file)
    pose_data = data['pose']  # (20, 17, 2) 형태

    # 각 프레임에서 skeleton이 정상적으로 인식된 keypoint 수 확인
    frame_success_counts = [np.sum(np.all(frame > 0, axis=-1)) for frame in pose_data]
    recognized_frames = sum(count > 0 for count in frame_success_counts)
    recognition_counts.append(recognized_frames)

# 히스토그램으로 성공률 시각화
plt.figure(figsize=(10, 6))
plt.hist(recognition_counts, bins=range(0, 21), edgecolor='black')
plt.title("Distribution of Successfully Recognized Frames per Video")
plt.xlabel("Number of Recognized Frames")
plt.ylabel("Number of Videos")
plt.grid(True, linestyle='--', alpha=0.6)

# 결과 저장
plt.savefig("/content/drive/MyDrive/skeleton_frame_success_histogram.png")
plt.show()

# 리스트 형태로도 결과 출력
recognition_summary = {i: recognition_counts.count(i) for i in range(21)}

print("\n--- Successfully Recognized Frames per Video ---")
for frame_count, video_count in recognition_summary.items():
    print(f"{frame_count} : {video_count} videos")

print("\nAnalysis complete. Histogram saved at /content/drive/MyDrive/skeleton_frame_success_histogram.png")