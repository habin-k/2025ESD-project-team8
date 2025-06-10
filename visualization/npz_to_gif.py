import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# COCO skeleton 연결 정보
skeleton = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12)
]

# npz 파일 로드
npz_file = Path("./npz_to_gif_sample_320.npz")
data = np.load(npz_file)
pose_data = data['pose']
label_data = data['label']

# ✅ 프레임 간 간격 계산 (총 10초)
total_duration_ms = 10_000  # 전체 재생 시간 (10초)
frame_count = len(pose_data)
interval_ms = total_duration_ms / frame_count  # 각 프레임 간 시간

fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(1, 0)
ax.set_aspect('equal')
scat = ax.scatter([], [], s=10, c='blue')

def update(frame):
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect('equal')
    ax.set_title(f"Frame {frame+1} - {'Fall' if label_data[frame] == 1 else 'No Fall'}")

    pose_frame = pose_data[frame]
    ax.scatter(pose_frame[:, 0], pose_frame[:, 1], s=10, c='red' if label_data[frame] == 1 else 'blue')

    for i, j in skeleton:
        if all(pose_frame[i] > 0) and all(pose_frame[j] > 0):
            ax.plot([pose_frame[i, 0], pose_frame[j, 0]], [pose_frame[i, 1], pose_frame[j, 1]],
                    'green' if label_data[frame] == 1 else 'blue')

ani = animation.FuncAnimation(fig, update, frames=frame_count, repeat=True, interval=interval_ms)
gif_path = "./npz_to_gif_sample.gif"
ani.save(gif_path, writer='pillow')

print(f"✅ GIF saved at {gif_path}, duration: 10 seconds, interval per frame: {interval_ms:.1f} ms")