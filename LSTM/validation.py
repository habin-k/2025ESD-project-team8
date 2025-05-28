import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

# ----- Dataset 로딩 -----
def load_all_npz_in_subfolders(root_dir):
    sequences, labels, lengths = [], [], []
    for path in Path(root_dir).rglob("*.npz"):
        data = np.load(path, allow_pickle=True)["arr_0"]
        if data.ndim == 3 and data.shape[1:] == (17, 3):
            reshaped = data[:, :, :2].reshape(data.shape[0], -1)  # (T, 34)
            sequences.append(torch.tensor(reshaped, dtype=torch.float32))
            labels.append(1)  # 낙상으로 고정
            lengths.append(len(reshaped))
    return sequences, labels, lengths

# ----- Dataset 클래스 -----
class FallDataset(Dataset):
    def __init__(self, sequences, labels, lengths):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lengths[idx]

# ----- Collate 함수 -----
def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(labels), torch.tensor(lengths)

# ----- LSTM 모델 -----
class PackedLSTMClassifier(nn.Module):
    def __init__(self, input_size=34, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        # 마지막 레이어의 forward + backward hidden state
        hn_cat = torch.cat((hn[-2], hn[-1]), dim=1)
        out = self.fc(hn_cat)
        return self.sigmoid(out).squeeze()

# ----- 평가 함수 -----
def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y, lengths in dataloader:
            pred = model(x, lengths)
            all_preds += (pred > 0.5).int().tolist()
            all_labels += y.int().tolist()
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return acc, f1

# ----- 메인 실행 -----
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dir = "./pose_tensor_npz/pose_tensor_npz/Y/Validation"
    
    print("Loading validation data...")
    val_sequences, val_labels, val_lengths = load_all_npz_in_subfolders(val_dir)
    print(f"Loaded {len(val_sequences)} sequences.")

    dataset = FallDataset(val_sequences, val_labels, val_lengths)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = PackedLSTMClassifier()
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)

    acc, f1 = evaluate(model, val_loader)
    print(f"[Validation] Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
