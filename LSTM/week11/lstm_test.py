import random
import matplotlib.pyplot as plt
import torch

# 1) Threshold와 샘플 개수 설정
THRESHOLD   = 0.8  # PR 커브에서 찾은 최적값
NUM_SAMPLES = 3

model = FrameLSTM()
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.cpu()
model.eval()

# 2) 검증 데이터 리스트에서 랜덤 인덱스 추출
#    val_seqs, val_labs, val_lens는 load_npz_frames()로 만들어진 리스트입니다.
indices = random.sample(range(len(val_seqs)), NUM_SAMPLES)

model.eval()
with torch.no_grad():
    for idx in indices:
        seq   = val_seqs[idx]   # (T,34)
        label = val_labs[idx]   # (T,)
        L     = val_lens[idx]   # T

        # 배치 차원 추가
        X       = seq.unsqueeze(0)         # (1, T, 34)
        lengths = torch.tensor([L])

        # 3) 모델 예측
        logits = model(X, lengths)         # (1, T_max)
        probs  = torch.sigmoid(logits)[0]  # (T_max,)
        preds  = (probs > THRESHOLD).int().tolist()
        trues  = label.int().tolist()

        # 4) 프레임별 정확도 계산
        correct = sum(p == t for p, t in zip(preds[:L], trues[:L]))
        frame_acc = correct / L

        # 5) 시각화
        T = L
        plt.figure(figsize=(8,2))
        # 확률 곡선
        plt.plot(probs[:T].cpu(), label='Pred Prob.')
        plt.fill_between(range(L), 0, 1, where=np.array(preds[:L])==1, color='blue', alpha=0.2, step='post')

        # 실제 라벨(0/1)은 계단식으로 표시
        plt.plot(trues[:T], label='True Label', alpha=0.6, drawstyle='steps-post')
        # 임계값선
        plt.axhline(THRESHOLD, color='gray', linestyle='--', label=f'Th={THRESHOLD}')
        plt.ylim(-0.1, 1.1)
        plt.xlim(0, T-1)
        plt.xlabel('Frame Index')
        plt.ylabel('Fall Prob. / Label')
        plt.title(f'Sample #{idx} (Length={T})  Frame Acc={frame_acc:.3f}')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
