import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# 1) 모델 로딩 직후에 CPU로 내리기
model = FrameLSTM()
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()  # 이제 모델도 CPU 모드

# 2) 그대로 val_loader 루프를 돌리기만 하면 됩니다.
all_probs, all_labels = [], []

with torch.no_grad():
    for X, Y, L in val_loader:
        logits = model(X, L)            # (B, T_max), 모두 CPU
        probs  = torch.sigmoid(logits)
        mask   = (torch.arange(probs.size(1))[None] < L[:, None])
        all_probs .extend(probs[mask].numpy().tolist())
        all_labels.extend(Y   [mask].numpy().tolist())

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels, dtype=int)

# -------------------------------------------------------------------
# 2) sklearn으로 PR 곡선 & AP 계산
precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
ap = average_precision_score(all_labels, all_probs)

# 최적 F1 임계값 찾기
f1_scores   = 2 * precision * recall / (precision + recall + 1e-8)
best_idx    = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]
best_f1     = f1_scores[best_idx]

print(f"Average Precision (AP): {ap:.4f}")
print(f"Best threshold: {best_thresh:.3f}, Max Frame‐level F1: {best_f1:.4f}")

# -------------------------------------------------------------------
# 3) 시각화
plt.figure(figsize=(6,6))
plt.step(recall, precision, where='post', label=f'AP={ap:.3f}')
plt.scatter(recall[best_idx], precision[best_idx],
            color='red',
            label=f'Best F1={best_f1:.3f}@Thr={best_thresh:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curve')
plt.xlim(0,1)
plt.ylim(0,1.05)
plt.legend()
plt.grid(True)
plt.show()
