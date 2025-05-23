import cv2
import torch
import numpy as np
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
cap = cv2.VideoCapture(0)  # or use a video file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO pose 추론
    results = yolo_model(frame)
    keypoints = results[0].keypoints

    if keypoints is not None and len(keypoints) > 0:
        # 첫 번째 사람만 추적
        kpts = keypoints.xy[0]  # shape: (17, 2)
        flat = kpts.cpu().numpy().flatten()  # (34,)

        queue.append(torch.tensor(flat, dtype=torch.float32))

        # Queue가 20개 모였을 때 추론
        if len(queue) == 20:
            x_seq = torch.stack(list(queue)).unsqueeze(0)  # (1, 20, 34)
            lengths = torch.tensor([20])
            with torch.no_grad():
                logits = lstm_model(x_seq, lengths)
                probs = torch.sigmoid(logits)

                # 현재 프레임 낙상 여부
                fall_prob = probs[0, -1].item()
                is_fall = fall_prob > THRESHOLD

                # 화면에 표시
                text = f"Fall: {'YES' if is_fall else 'NO'} ({fall_prob:.2f})"
                cv2.putText(frame, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if is_fall else (0, 255, 0), 2)

    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
