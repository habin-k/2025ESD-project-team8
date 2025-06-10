import cv2
import torch
import numpy as np
import time
from collections import deque
from openvino.runtime import Core
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import matplotlib.pyplot as plt

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

# --------- LSTM 모델 로딩 ---------
lstm_model = FrameLSTM()
lstm_model.load_state_dict(torch.load("best_model.pth", map_location='cpu'))
lstm_model.eval()

# --------- OpenVINO 모델 로딩 ---------
core = Core()
model_ov = core.read_model("yolov8n-pose_openvino/yolov8n-pose.xml")
compiled_model = core.compile_model(model_ov, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# --------- COCO Skeleton 연결 정보 ---------
skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (11, 12)
]

# --------- 웹캠 설정 ---------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("❌ 웹캠을 열 수 없습니다.")

# --------- 초기화 ---------
THRESHOLD = 0.8
queue = deque(maxlen=20)
queue_frame_indices = deque(maxlen=20)
zero_vector = torch.zeros(51, dtype=torch.float32)
frame_index = 0
prev_time = time.time()

def normalize_keypoints(kpts, conf, width, height):
    kpts_np = np.copy(kpts)
    kpts_np[:, 0] /= width
    kpts_np[:, 1] /= height
    combined = np.concatenate([kpts_np, conf[:, None]], axis=-1)
    return torch.tensor(combined.flatten(), dtype=torch.float32)

# --------- 실시간 Plot ---------
plt.ion()
fig, ax = plt.subplots()
line_prob, = ax.plot([], [], label="Fall Prob.", color="dodgerblue")
line_label, = ax.plot([], [], label="Fall Label", color="orange")
ax.axhline(THRESHOLD, linestyle="--", color="gray")
ax.set_ylim(0, 1)
ax.set_xlim(0, 19)
ax.set_xlabel("Queue Index")
ax.set_ylabel("Probability")
ax.set_title("Live LSTM Prediction")
ax.legend()

print("[INFO] 실시간 추론 시작 ('q' 키로 종료)")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 프레임 읽기 실패")
            continue

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        input_img = cv2.resize(frame, (640, 640))
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

            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 640
            kpts[:, 0] *= scale_x
            kpts[:, 1] *= scale_y

            cx, cy, w, h = best[0:4]
            x1 = int((cx - w / 2) * scale_x)
            y1 = int((cy - h / 2) * scale_y)
            x2 = int((cx + w / 2) * scale_x)
            y2 = int((cy + h / 2) * scale_y)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"person {max_conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            for i, (pt, c) in enumerate(zip(kpts, confs)):
                if c > 0.2:
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            for a, b in skeleton:
                if confs[a] > 0.2 and confs[b] > 0.2:
                    pt1 = tuple(kpts[a].astype(int))
                    pt2 = tuple(kpts[b].astype(int))
                    cv2.line(frame, pt1, pt2, (255, 255, 0), 2)

            norm_vector = normalize_keypoints(kpts, confs, frame.shape[1], frame.shape[0])
        else:
            norm_vector = zero_vector

        queue.append(norm_vector)
        queue_frame_indices.append(frame_index)
        frame_index += 1

        fall_prob = 0.0
        fall_label = "No Fall"

        if len(queue) == 20:
            x_seq = torch.stack(list(queue)).unsqueeze(0)
            lengths = torch.tensor([20])
            with torch.no_grad():
                logits = lstm_model(x_seq, lengths)
                probs = torch.sigmoid(logits)[0].cpu().numpy()

                fall_prob = float(probs[-1])
                fall_label = "Fall" if fall_prob > THRESHOLD else "No Fall"

                x = list(range(20))
                y = probs
                y_bin = (y > THRESHOLD).astype(int)

                line_prob.set_ydata(y)
                line_prob.set_xdata(x)
                line_label.set_ydata(y_bin)
                line_label.set_xdata(x)
                ax.set_title(f"LSTM Prediction [{queue_frame_indices[0]}~{queue_frame_indices[-1]}]")
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()

        # --------- OpenCV 표시 ---------
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Fall Prob: {fall_prob:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Label: {fall_label}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if fall_label == "Fall" else (0, 255, 0), 2)

        cv2.imshow("Webcam with Skeleton + Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 'q' 입력으로 종료합니다.")
            break

except KeyboardInterrupt:
    print("\n[INFO] Ctrl+C 입력으로 종료합니다.")

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()
