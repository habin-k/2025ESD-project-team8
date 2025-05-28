import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import matplotlib.pyplot as plt
import os

# --------- 1. 모델 로딩 ---------
yolo_model = YOLO('yolov8n-pose.pt')

class FrameLSTM(torch.nn.Module):
    def __init__(self, input_size=34, hidden_size=64):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size,
                                  num_layers=2, batch_first=True,
                                  bidirectional=True, dropout=0.3)
        self.fc = torch.nn.Linear(hidden_size * 2, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out_unpad, _ = pad_packed_sequence(out_packed, batch_first=True)
        return self.fc(out_unpad).squeeze(-1)

lstm_model = FrameLSTM()
lstm_model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
lstm_model.eval()

# --------- 2. 설정 ---------
video_path = "00015_H_A_SY_C2.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
interval_frame = int(fps * 0.5)

queue = deque(maxlen=20)
THRESHOLD = 0.75
frame_index = 0
frame_count = 0

fall_probs = []
fall_flags = []
fall_frame_indices = []

# --------- 3. 큐 초기화: 첫 skeleton이 나올 때까지 대기 ---------
print("[INFO] Waiting for first valid skeleton to initialize queue...")

first_vector = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("❌ Could not find a valid skeleton in the video.")

    results = yolo_model(frame)
    keypoints = results[0].keypoints

    if keypoints is not None and len(keypoints) > 0:
        try:
            kpts = keypoints.xy[0]
            flat = kpts.cpu().numpy().flatten()
            if flat.shape[0] == 34:
                first_vector = torch.tensor(flat, dtype=torch.float32)
                for _ in range(20):
                    queue.append(first_vector)
                print("[INFO] Queue initialized with first valid skeleton.")
                break
        except:
            continue

# --------- 4. 이어서 추론 수행 ---------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % interval_frame == 0:
        results = yolo_model(frame)
        keypoints = results[0].keypoints

        if keypoints is not None and len(keypoints) > 0:
            try:
                kpts = keypoints.xy[0]
                flat = kpts.cpu().numpy().flatten()
                if flat.shape[0] == 34:
                    queue.append(torch.tensor(flat, dtype=torch.float32))
            except:
                continue
        else:
            queue.append(first_vector)  # 여전히 사람 없으면 이전 skeleton 유지

        if len(queue) == 20:
            x_seq = torch.stack(list(queue)).unsqueeze(0)
            lengths = torch.tensor([20])
            with torch.no_grad():
                logits = lstm_model(x_seq, lengths)
                probs = torch.sigmoid(logits)
                fall_prob = probs[0, -1].item()
                is_fall = fall_prob > THRESHOLD

                fall_probs.append(fall_prob)
                fall_flags.append(int(is_fall))
                fall_frame_indices.append(frame_index)

                print(f"[{frame_index}] Fall: {'YES' if is_fall else 'NO'} ({fall_prob:.2f})")

            frame_index += 1

    frame_count += 1

cap.release()

# --------- 5. 그래프 출력 ---------
if len(fall_frame_indices) > 0:
    plt.figure(figsize=(10, 4))
    plt.plot(fall_frame_indices, fall_probs, label="Fall Probability", marker='o')
    plt.axhline(THRESHOLD, color='r', linestyle='--', label=f"Threshold = {THRESHOLD}")
    plt.xlabel("Frame Index")
    plt.ylabel("Fall Probability")
    plt.title("Fall Detection over Time")
    plt.xticks(range(min(fall_frame_indices), max(fall_frame_indices)+1, 1))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fall_detection_plot.png")
    plt.show()
else:
    print("[Warning] No valid inference results to plot.")
