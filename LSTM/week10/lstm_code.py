import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import random

# 1. 모든 Training 폴더 내 .npz 파일 불러오기 (재귀 탐색)
def load_npz_in_training_folders(root_dir, label):
    sequences, labels, lengths = [], [], []
    found_files = 0  # 파일 개수 카운트
    
    for subdir, dirs, files in os.walk(root_dir):
        # Training 폴더 내부의 모든 .npz 파일 탐색
        if 'Training' in subdir.split(os.sep):
            for fname in files:
                if fname.endswith(".npz"):
                    npz_path = os.path.join(subdir, fname)
                    
                    try:
                        # npz 파일 내 모든 배열 로딩
                        npz_data = np.load(npz_path, allow_pickle=True)
                        
                        # "pose" 데이터만 로딩
                        if "pose" in npz_data.files:
                            data = npz_data["pose"]
                            
                            # (20, 17, 2) 형식의 pose 데이터만 로딩
                            if data.ndim == 3 and data.shape[1:] == (17, 2):
                                # (20, 34)로 변환
                                data = data.reshape(data.shape[0], -1)
                                
                                # (0,0) 좌표 예외 처리
                                zero_mask = (data == 0).all(axis=1)
                                for t in range(1, data.shape[0]):
                                    if zero_mask[t]:  # 현재 프레임이 전부 (0,0)일 때
                                        data[t] = data[t-1]  # 이전 프레임으로 복사

                                # 텐서로 변환 후 리스트에 추가
                                sequences.append(torch.tensor(data, dtype=torch.float32))
                                labels.append(label)
                                lengths.append(data.shape[0])
                                found_files += 1

                    except Exception as e:
                        print(f"파일 로드 실패: {npz_path}, 오류: {e}")

    print(f"총 {found_files}개의 {label} 데이터 로딩 완료")
    return sequences, labels, lengths

# 2. 경로 설정
fall_dir = "/content/pt/pose_tensor_npz/Y/Training"
normal_dir = "/content/pt/pose_tensor_npz/N/Training"

fall_seqs, fall_labs, fall_lens = load_npz_in_training_folders(fall_dir, 1)
normal_seqs, normal_labs, normal_lens = load_npz_in_training_folders(normal_dir, 0)

# 3. 데이터 결합 및 셔플
sequences = fall_seqs + normal_seqs
labels = fall_labs + normal_labs
lengths = fall_lens + normal_lens

if len(sequences) == 0 or len(labels) == 0 or len(lengths) == 0:
    print("데이터가 비어 있습니다. 경로 또는 파일 형식을 확인하세요.")
else:
    combined = list(zip(sequences, labels, lengths))
    random.shuffle(combined)
    sequences, labels, lengths = zip(*combined)
    print(f"총 {len(sequences)}개의 시퀀스 로딩 완료")

# 4. Dataset 정의
class FallDataset(Dataset):
    def __init__(self, sequences, labels, lengths):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lengths[idx]

# 5. Collate 함수 (패딩)
def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(labels), torch.tensor(lengths)

# 6. DataLoader 준비
dataset = FallDataset(sequences, labels, lengths)
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)


# 7. LSTM 모델 정의
class PackedLSTMClassifier(nn.Module):
    def __init__(self, input_size=34, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        out = self.fc(hn[-1])
        return self.sigmoid(out).squeeze()

model = PackedLSTMClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()
loss_history = []

# 8. 학습 루프
for epoch in range(10):
    all_preds, all_labels = [], []
    for x, y, lengths in loader:
        # 모델 예측
        pred = model(x, lengths)
        # 손실 계산
        loss = loss_fn(pred, y)
        # 역전파
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_history.append(loss.item())
        
        # 예측 결과 저장
        all_preds += (pred > 0.5).int().tolist()
        all_labels += y.int().tolist()

    # 정확도와 F1 점수 계산
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

# 9. 손실 시각화
plt.plot(loss_history)
plt.title("Training Loss (Fall vs Normal)")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
