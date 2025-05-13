import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import json
import random
from tqdm.notebook import tqdm  # Colab 환경에서 tqdm 사용

# ✅ CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# 1. 모든 Training 폴더 내 .npz 파일 불러오기 (재귀 탐색)
def load_npz_in_training_folders(root_dir):
    sequences, labels, lengths = [], [], []
    found_files = 0
    
    for subdir, _, files in os.walk(root_dir):
        if 'Training' in subdir.split(os.sep):
            for fname in tqdm(files, desc="🔍 Loading npz files", leave=False):
                if fname.endswith(".npz"):
                    npz_path = os.path.join(subdir, fname)
                    
                    try:
                        npz_data = np.load(npz_path, allow_pickle=True)
                        if "pose" in npz_data.files and "label" in npz_data.files:
                            data = npz_data["pose"]
                            label = npz_data["label"]
                            
                            if data.ndim == 3 and data.shape[1:] == (17, 2):
                                data = data.reshape(data.shape[0], -1)
                                zero_mask = (data == 0).all(axis=1)
                                for t in range(1, data.shape[0]):
                                    if zero_mask[t]:
                                        data[t] = data[t-1]

                                sequences.append(torch.tensor(data, dtype=torch.float32))
                                labels.append(torch.tensor(label, dtype=torch.float32))
                                lengths.append(data.shape[0])
                                found_files += 1

                    except Exception as e:
                        print(f"파일 로드 실패: {npz_path}, 오류: {e}")

    print(f"✅ 총 {found_files}개의 데이터 로딩 완료")
    return sequences, labels, lengths

# 2. 경로 설정
fall_dir = "/content/drive/MyDrive/pose_tensor_npz/Y/Training"
normal_dir = "/content/drive/MyDrive/pose_tensor_npz/N/Training"

fall_seqs, fall_labs, fall_lens = load_npz_in_training_folders(fall_dir)
normal_seqs, normal_labs, normal_lens = load_npz_in_training_folders(normal_dir)

# 3. 데이터 결합 및 셔플
sequences = fall_seqs + normal_seqs
labels = fall_labs + normal_labs
lengths = fall_lens + normal_lens
combined = list(zip(sequences, labels, lengths))
random.shuffle(combined)
sequences, labels, lengths = zip(*combined)

# 4. Dataset 정의
class FallDataset(Dataset):
    def __init__(self, sequences, labels, lengths):
        self.sequences = sequences
        self.labels = labels
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lengths[idx]

# 5. Collate 함수 (패딩)
def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    padded_seq = pad_sequence(sequences, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True)
    return padded_seq, padded_labels, torch.tensor(lengths)

# 6. DataLoader 준비
dataset = FallDataset(sequences, labels, lengths)
loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# 7. LSTM 모델 정의 (프레임별 예측)
class PackedLSTMClassifier(nn.Module):
    def __init__(self, input_size=34, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_size, 1).to(device)
        self.sigmoid = nn.Sigmoid().to(device)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hn, _) = self.lstm(packed)
        unpacked_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # 각 프레임마다 예측
        out = self.sigmoid(self.fc(unpacked_output))  # (Batch, Sequence Length, 1)
        return out.squeeze(-1)  # (Batch, Sequence Length)

# 모델 초기화
model = PackedLSTMClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()
loss_history = []

# 8. 학습 루프 (프레임별 예측)
for epoch in range(10):
    all_preds, all_labels = [], []
    model.train()
    
    with tqdm(loader, desc=f"🔧 Training Epoch {epoch+1}", leave=False) as pbar:
        for x, y, lengths in pbar:
            x, y = x.to(device), y.to(device)
            pred = model(x, lengths)  # (Batch, Sequence Length)
            
            # 손실 계산 (프레임별 Binary Cross Entropy)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_history.append(loss.item())

            all_preds += (pred > 0.5).int().flatten().tolist()
            all_labels += y.int().flatten().tolist()
            pbar.set_postfix({"Loss": loss.item()})

    # 정확도와 F1 점수 계산 (프레임 단위)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"✅ Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

# 9. 학습 결과 저장
save_dir = "/content/drive/MyDrive/lstm_training_results"
torch.save(model.state_dict(), os.path.join(save_dir, "best_lstm_model.pth"))

results = {
    "loss_history": loss_history,
    "final_accuracy": acc,
    "final_f1_score": f1
}

with open(os.path.join(save_dir, "training_results.json"), "w") as f:
    json.dump(results, f)

print(f"✅ 학습 결과 저장 완료.")
