import cv2
import torch
import numpy as np
import time
from collections import deque
from openvino.runtime import Core
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

# --------- 설정 ---------
target_fps = 5  # 원하는 FPS (예: 5 FPS → 0.2초 간격)
frame_interval = 1.0 / target_fps  # 프레임 간 최소 간격 (초)

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

# --------- Letterbox 함수 ---------
def letterbox_image(image, size=(320, 320)):
    h, w = image.shape[:2]
    scale = min(size[0] / w, size[1] / h)
    nw, nh = int(w * scale), int(h * scale)
    image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    top = (size[1] - nh) // 2
    left = (size[0] - nw) // 2
    new_image[top:top+nh, left:left+nw] = image_resized
    return new_image, scale, left, top

# --------- 모델 로딩 ---------
lstm_model = FrameLSTM()
lstm_model.load_state_dict(torch.load("best_model.pth", map_location='cpu'))
lstm_model.eval()

core = Core()
model_ov = core.read_model("yolo11n-pose_openvino_320/yolo11n-pose.xml")
compiled_model = core.compile_model(model_ov, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# --------- 캡처 설정 ---------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("❌ 웹캠을 열 수 없습니다.")

# --------- 초기화 ---------
queue = deque(maxlen=20)
zero_vector = torch.zeros(51, dtype=torch.float32)
prev_time = time.time()

def normalize_keypoints(kpts, conf, width, height):
    kpts_np = np.copy(kpts)
    kpts_np[:, 0] /= width
    kpts_np[:, 1] /= height
    combined = np.concatenate([kpts_np, conf[:, None]], axis=-1)
    return torch.tensor(combined.flatten(), dtype=torch.float32)

print(f"[INFO] 실시간 추론 시작 ('q' 키로 종료, {target_fps} FPS 제한)")

try:
    while True:
        loop_start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            continue

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        # Letterbox 전처리
        input_img, scale, pad_x, pad_y = letterbox_image(frame, size=(320, 320))
        input_tensor = input_img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0

        # YOLO 추론
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
            kpts[:, 0] = (kpts[:, 0] - pad_x) / scale
            kpts[:, 1] = (kpts[:, 1] - pad_y) / scale
            norm_vector = normalize_keypoints(kpts, confs, frame.shape[1], frame.shape[0])
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

        # FPS 제한
        inference_time = time.time() - loop_start_time
        if inference_time < frame_interval:
            time.sleep(frame_interval - inference_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 'q' 입력으로 종료합니다.")
            break

except KeyboardInterrupt:
    print("\n[INFO] Ctrl+C 입력으로 종료합니다.")

cap.release()
cv2.destroyAllWindows()
