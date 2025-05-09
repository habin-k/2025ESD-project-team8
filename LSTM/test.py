import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import random

# 1. 데이터 로딩 함수
def load_all_npy_in_subfolders(root_dir, label):
    """
    지정된 디렉토리에서 모든 npy 파일을 로드하고, 
    각 파일을 3차원 텐서 (T, 17, 2)로 간주하여 
    2차원 (T, 34)로 변환하여 시퀀스로 저장.

    Args:
        root_dir (str): npy 파일이 저장된 폴더 경로
        label (int): 해당 시퀀스의 라벨 (낙상: 1, 정상: 0)

    Returns:
        sequences (list of Tensors): 각 파일의 프레임 시퀀스 (T, 34)
        labels (list): 각 시퀀스의 라벨
        lengths (list): 각 시퀀스의 길이 (프레임 수)
    """
    sequences, labels, lengths = [], [], []
    for subdir, _, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith(".npy"):
                path = os.path.join(subdir, fname)
                data = np.load(path, allow_pickle=True)
                if data.ndim == 3 and data.shape[1:] == (17, 2):
                    reshaped = data.reshape(data.shape[0], -1)  # (T, 34)
                    sequences.append(torch.tensor(reshaped, dtype=torch.float32))
                    labels.append(label)
                    lengths.append(len(reshaped))
    return sequences, labels, lengths

# 2. 경로 설정 (낙상: 1, 정상: 0)
fall_dir = "/content/pose_tensor/Y"
normal_dir = "/content/pose_tensor/N"

fall_seqs, fall_labs, fall_lens = load_all_npy_in_subfolders(fall_dir, 1)
normal_seqs, normal_labs, normal_lens = load_all_npy_in_subfolders(normal_dir, 0)

# 3. 데이터셋 나누기 (8:2 비율)
from sklearn.model_selection import train_test_split
sequences = fall_seqs + normal_seqs
labels = fall_labs + normal_labs
lengths = fall_lens + normal_lens

X_train, X_val, y_train, y_val, len_train, len_val = train_test_split(
    sequences, labels, lengths, test_size=0.2, random_state=42, stratify=labels)

# 4. Dataset 클래스
class FallDataset(Dataset):
    def __init__(self, sequences, labels, lengths):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lengths[idx]

# 5. Collate 함수
def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(labels), torch.tensor(lengths)

# 6. 모델 정의
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

# 7. 학습 및 검증 설정
train_dataset = FallDataset(X_train, y_train, len_train)
val_dataset = FallDataset(X_val, y_val, len_val)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

model = PackedLSTMClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# 8. 학습 루프
for epoch in range(30):
    model.train()
    train_loss = 0
    for x, y, lengths in train_loader:
        pred = model(x, lengths)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 검증 루프
    model.eval()
    val_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for x, y, lengths in val_loader:
            pred = model(x, lengths)
            loss = loss_fn(pred, y)
            val_loss += loss.item()
            all_preds += (pred > 0.5).int().tolist()
            all_labels += y.int().tolist()

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
