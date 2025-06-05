import cv2
import torch
import numpy as np
import time
from collections import deque
from openvino.runtime import Core
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

core = Core()
model_ov = core.read_model("yolov8n-pose_openvino_320/yolov8n-pose.xml")
compiled_model = core.compile_model(model_ov, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# --------- 캡처 설정 ---------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("❌ 웹캠을 열 수 없습니다.")

# --------- 초기화 ---------
THRESHOLD = 0.8
queue = deque(maxlen=20)
zero_vector = torch.zeros(51, dtype=torch.float32)
prev_time = None

def normalize_keypoints(kpts, conf, width, height):
    kpts_np = np.copy(kpts)
    kpts_np[:, 0] /= width
    kpts_np[:, 1] /= height
    combined = np.concatenate([kpts_np, conf[:, None]], axis=-1)
    return torch.tensor(combined.flatten(), dtype=torch.float32)

print("[INFO] 실시간 추론 시작 ('q' 키로 종료)")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time else 0.0
        prev_time = curr_time

        input_img = cv2.resize(frame, (320, 320))
        input_tensor = input_img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0

        outputs = compiled_model([input_tensor])
        output_tensor = outputs[output_layer]
        results = np.squeeze(output_tensor).transpose(1, 0)

        best = None
        max_conf = 0.25

        for det in results:
            if det.shape[0] != 56:
                continue
            conf = det[4]
            if conf > max_conf:
                best = det
                max_conf = conf

        if best is not None:
            kpt_raw = best[5:]
            kpts = kpt_raw.reshape(-1, 3)[:, :2]
            confs = kpt_raw.reshape(-1, 3)[:, 2]
            norm_vector = normalize_keypoints(kpts, confs, 320, 320)
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