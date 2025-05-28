import cv2
import torch
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# --------- 1. YOLOv8n-pose 불러오기 ---------
yolo_model = YOLO('yolov8n-pose.pt')

# --------- 2. 학습된 LSTM 모델 구조 복원 ---------
class FrameLSTM(torch.nn.Module):
    def __init__(self, input_size=34, hidden_size=64):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size,
                                  num_layers=2, batch_first=True,
                                  bidirectional=True, dropout=0.3)
        self.fc = torch.nn.Linear(hidden_size*2, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out_unpad, _ = pad_packed_sequence(out_packed, batch_first=True)
        return self.fc(out_unpad).squeeze(-1)

lstm_model = FrameLSTM()
lstm_model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
lstm_model.eval()

# --------- 3. Skeleton queue ---------
queue = deque(maxlen=20)
THRESHOLD = 0.75

# --------- 4. 웹캠 열기 ---------
cap = cv2.VideoCapture(0)

# --------- 5. FPS 설정 ---------
TARGET_FPS = 2
TARGET_FRAME_TIME = 1.0 / TARGET_FPS

while cap.isOpened():
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # YOLO pose 추론
    results = yolo_model(frame)
    keypoints = results[0].keypoints

    # --------- 예외처리: keypoints가 None이거나 빈 경우 ---------
    if keypoints is None or len(keypoints) == 0:
        cv2.putText(frame, "No person detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    else:
        try:
            kpts = keypoints.xy[0]  # (17, 2)
            flat = kpts.cpu().numpy().flatten()  # (34,)
            if flat.shape[0] == 34:
                queue.append(torch.tensor(flat, dtype=torch.float32))
        except Exception as e:
            print(f"[Warning] Failed to extract keypoints: {e}")

        # 추론 조건 만족 시 실행
        if len(queue) == 20:
            try:
                x_seq = torch.stack(list(queue)).unsqueeze(0)  # (1, 20, 34)
                lengths = torch.tensor([20])
                with torch.no_grad():
                    logits = lstm_model(x_seq, lengths)
                    probs = torch.sigmoid(logits)

                    fall_prob = probs[0, -1].item()
                    is_fall = fall_prob > THRESHOLD

                    text = f"Fall: {'YES' if is_fall else 'NO'} ({fall_prob:.2f})"
                    cv2.putText(frame, text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255) if is_fall else (0, 255, 0), 2)
            except Exception as e:
                print(f"[Error] Inference failed: {e}")

    # --------- FPS 계산 및 표시 ---------
    fps = 1 / (time.time() - start_time + 1e-8)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Fall Detection", frame)

    # --------- 2 FPS 고정 (0.5초 간격) ---------
    elapsed = time.time() - start_time
    time_to_sleep = TARGET_FRAME_TIME - elapsed
    if time_to_sleep > 0:
        time.sleep(time_to_sleep)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
