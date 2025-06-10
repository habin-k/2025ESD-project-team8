import cv2
import torch
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

# --------- LSTM 모델 정의 ---------
class FrameLSTM(torch.nn.Module):
    def __init__(self, input_size=51, hidden_size=64):
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

# --------- 모델 로딩 ---------
lstm_model = FrameLSTM()
lstm_model.load_state_dict(torch.load("best_model.pth", map_location='cpu'))
lstm_model.eval()

yolo_model = YOLO('yolov8n-pose.pt')

# --------- 웹캠 설정 ---------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("❌ 웹캠을 열 수 없습니다.")

frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# --------- 큐 및 변수 초기화 ---------
THRESHOLD = 0.8
queue = deque(maxlen=20)
zero_vector = torch.zeros(51, dtype=torch.float32)
prev_time = time.time()

def normalize_keypoints(kpts_tensor, width, height, conf_tensor):
    kpts_np = kpts_tensor.cpu().numpy()
    conf_np = conf_tensor.cpu().numpy()
    kpts_np[:, 0] /= width
    kpts_np[:, 1] /= height
    combined = np.concatenate([kpts_np, conf_np[:, None]], axis=-1)
    return torch.tensor(combined.flatten(), dtype=torch.float32)

print("[INFO] 실시간 추론 시작 ('q' 키로 종료)")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # FPS 계산
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        results = yolo_model(frame)
        keypoints = results[0].keypoints

        if keypoints is not None and len(keypoints) > 0:
            try:
                kpts = keypoints.xy[0]
                conf = keypoints.conf[0]
                if kpts.shape == (17, 2) and conf.shape == (17,):
                    norm_vector = normalize_keypoints(kpts, frame_width, frame_height, conf)
                else:
                    norm_vector = zero_vector
            except:
                norm_vector = zero_vector
        else:
            norm_vector = zero_vector

        queue.append(norm_vector)

        fall_prob = 0.0
        if len(queue) == 20:
            x_seq = torch.stack(list(queue)).unsqueeze(0)
            lengths = torch.tensor([20])
            with torch.no_grad():
                logits = lstm_model(x_seq, lengths)
                probs = torch.sigmoid(logits)[0].cpu().numpy()
                fall_prob = float(probs[-1])

        print(f"FPS: {fps:.2f}, Fall Probability: {fall_prob:.3f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 'q' 입력으로 종료합니다.")
            break

except KeyboardInterrupt:
    print("\n[INFO] Ctrl+C 입력으로 종료합니다.")

cap.release()
cv2.destroyAllWindows()
