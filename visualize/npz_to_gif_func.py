import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

def npz_to_pose_gif(npz_path, gif_save_path, normalized=False, image_width=640, image_height=384):
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
    data = np.load(npz_path)
    pose_data = data['pose']        # (20, 17, 2)
    label_data = data['label']      # (20,)

    if normalized:
        pose_data = pose_data.copy()
        pose_data[:, :, 0] /= image_width
        pose_data[:, :, 1] /= image_height

    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # y축 반전
        ax.set_aspect('equal')
        ax.set_title(f"Frame {frame+1} - {'Fall' if label_data[frame] == 1 else 'No Fall'}")

        pose = pose_data[frame]
        color = 'red' if label_data[frame] == 1 else 'blue'

        # 마커 그리기
        ax.scatter(pose[:, 0], pose[:, 1], s=10, c=color, zorder=2)

        # 연결선 그리기
        for i, j in skeleton:
            if np.all(pose[i] > 0) and np.all(pose[j] > 0):
                ax.plot([pose[i, 0], pose[j, 0]],
                        [pose[i, 1], pose[j, 1]],
                        color='green' if label_data[frame] == 1 else 'blue',
                        linewidth=1.5,
                        zorder=1)

    ani = animation.FuncAnimation(fig, update, frames=len(pose_data), repeat=True, interval=500)
    ani.save(gif_save_path, writer='pillow')
    print(f"✅ GIF saved at: {gif_save_path}")


npz_to_pose_gif(
    npz_path="./00015_H_A_SY_C2.npz",
    gif_save_path="./00015_H_A_SY_C2.gif",
    normalized=True,               # True면 정규화된 npz 파일일 때
    image_width=640,
    image_height=384
)