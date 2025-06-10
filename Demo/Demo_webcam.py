import cv2
import torch
import numpy as np
import time
from collections import deque
from openvino.runtime import Core
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import matplotlib.pyplot as plt
import RPi.GPIO as GPIO


BUZZER_PIN = 4  # BCM 기준
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# PWM 초기화 (주파수 1000Hz)
pwm = GPIO.PWM(BUZZER_PIN, 1000)


# 부저 울림 제어용 변수
buzzer_last_on = 0
buzzer_cooldown = 5  # 반복 울림 방지를 위한 쿨다운 (초)


# --------- FPS 제한 설정 ---------
target_fps = 5
frame_interval = 1.0 / target_fps

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

# --------- COCO Skeleton ---------
skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (11, 12)
]

# --------- 캡처 설정 ---------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("❌ 웹캠을 열 수 없습니다.")

# --------- 초기화 ---------
THRESHOLD = 0.9
queue = deque(maxlen=50)
queue_frame_indices = deque(maxlen=50)
zero_vector = torch.zeros(51, dtype=torch.float32)
frame_index = 0
prev_time = time.time()

def normalize_keypoints(kpts, conf, width=320, height=320):
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
ax.set_xlim(0, 49)
ax.set_xlabel("Queue Index")
ax.set_ylabel("Probability")
ax.set_title("Live LSTM Prediction")
ax.legend()

print(f"[INFO] 실시간 추론 시작 ('q' 키로 종료, {target_fps} FPS 제한)")

try:
    while True:
        loop_start_time = time.time()

        ret, frame = cap.read()  # 이 프레임만 사용
        if not ret:
            print("[WARN] 프레임 읽기 실패")
            break  # 영상 끝나면 종료

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
        max_conf = 0.1

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

            # 🔧 마스킹 처리
            conf_threshold = 0.2
            low_conf_mask = confs < conf_threshold
            kpts[low_conf_mask] = 0.0
            confs[low_conf_mask] = 0.0

            kpts[:, 0] = (kpts[:, 0] - pad_x) / scale
            kpts[:, 1] = (kpts[:, 1] - pad_y) / scale

            cx, cy, w, h = best[0:4]
            x1 = int((cx - w / 2) * 320)
            y1 = int((cy - h / 2) * 320)
            x2 = int((cx + w / 2) * 320)
            y2 = int((cy + h / 2) * 320)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"person {max_conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            for i, (pt, c) in enumerate(zip(kpts, confs)):
                if c > conf_threshold:
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            for a, b in skeleton:
                if confs[a] > conf_threshold and confs[b] > conf_threshold:
                    pt1 = tuple(kpts[a].astype(int))
                    pt2 = tuple(kpts[b].astype(int))
                    cv2.line(frame, pt1, pt2, (255, 255, 0), 2)
                    
            # 관절 수 확인 (conf > conf_threshold인 관절 수)
            valid_joint_count = np.sum(confs > conf_threshold)

            if valid_joint_count < 5:
                norm_vector = zero_vector
            else:
                norm_vector = normalize_keypoints(kpts, confs, 320, 320)
        else:
            norm_vector = zero_vector

        queue.append(norm_vector)
        queue_frame_indices.append(frame_index)
        frame_index += 1

        fall_prob = 0.0
        fall_label = "No Fall"

        if len(queue) == 50:
            x_seq = torch.stack(list(queue)).unsqueeze(0)
            lengths = torch.tensor([50])
            with torch.no_grad():
                logits = lstm_model(x_seq, lengths)
                probs = torch.sigmoid(logits)[0].cpu().numpy()
                

                fall_prob = float(probs[-1])
                fall_label = "Fall" if fall_prob > THRESHOLD else "No Fall"
                
                # 예시: 낙상 감지 시 처리 로직 안에 넣기
                if fall_prob > THRESHOLD:
                    now = time.time()
                    if now - buzzer_last_on > buzzer_cooldown:
                        print("[BUZZER] 낙상 감지됨! 3초간 부저 울림")
                        pwm.start(20)        # duty cycle 50%로 시작
                        time.sleep(1)        # 3초 울림
                        pwm.stop()           # 부저 끔
                        buzzer_last_on = now

                x = list(range(49))
                y = probs[1:]
                y_bin = (y > THRESHOLD).astype(int)

                line_prob.set_data(x, y)
                line_label.set_data(x, y_bin)
                ax.set_title(f"LSTM Prediction [{queue_frame_indices[0]}~{queue_frame_indices[-1]}]")
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Fall Prob: {fall_prob:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Label: {fall_label}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if fall_label == "Fall" else (0, 255, 0), 2)

        cv2.imshow("Webcam with Skeleton + Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 'q' 입력으로 종료합니다.")
            break

        # --------- FPS 제한 ---------
        inference_time = time.time() - loop_start_time
        if inference_time < frame_interval:
            time.sleep(frame_interval - inference_time)

except KeyboardInterrupt:
    print("\n[INFO] Ctrl+C 입력으로 종료합니다.")

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()
GPIO.cleanup()