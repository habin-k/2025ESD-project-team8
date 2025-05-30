import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import matplotlib.pyplot as plt
import random

# ----- Dataset 로딩 -----
def load_all_npz_in_subfolders(root_dir):
    sequences, labels, lengths, file_names = [], [], [], []
    npz_paths = list(Path(root_dir).rglob("*.npz"))
    print(f"🔍 Found {len(npz_paths)} .npz files in '{root_dir}'")

    for path in npz_paths:
        try:
            npz_data = np.load(path, allow_pickle=True)
            if "pose" in npz_data and "label" in npz_data:
                pose_data = npz_data["pose"]
                label_data = npz_data["label"]
                if pose_data.shape == (20, 17, 2) and label_data.shape == (20,):
                    reshaped = pose_data.reshape(20, -1)  # (20, 34)
                    sequences.append(torch.tensor(reshaped, dtype=torch.float32))
                    labels.append(torch.tensor(label_data, dtype=torch.float32))
                    lengths.append(20)
                    file_names.append(path.stem)
        except Exception as e:
            print(f"❌ Error loading {path}: {e}")
    return sequences, labels, lengths, file_names

# ----- Dataset 클래스 -----
class FallDataset(Dataset):
    def __init__(self, sequences, labels, lengths, file_names):
        self.sequences = sequences
        self.labels = labels
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.file_names = file_names

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lengths[idx], self.file_names[idx]

# ----- Collate 함수 -----
def collate_fn(batch):
    sequences, labels, lengths, file_names = zip(*batch)
    padded_seq = pad_sequence(sequences, batch_first=True)     # (B, T, 34)
    padded_label = pad_sequence(labels, batch_first=True)       # (B, T)
    return padded_seq, padded_label, torch.tensor(lengths), file_names

# ----- LSTM 모델 -----
class FrameLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        unpacked, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, T, 2H)
        logits = self.fc(unpacked).squeeze(-1)  # (B, T)
        return self.sigmoid(logits)

# ----- 시각화 함수 -----
def plot_frame_predictions(preds, labels, video_name="video"):
    plt.figure(figsize=(10, 3))
    plt.plot(preds, label='Prediction', marker='o')
    plt.plot(labels, label='Ground Truth', marker='x')
    plt.title(f"Frame-wise Fall Prediction - {video_name}")
    plt.xlabel("Frame")
    plt.ylabel("Fall (1) / No Fall (0)")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def visualize_random_samples(model, dataset, device, num_samples=3):
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)

    for i, idx in enumerate(indices):
        x, y, length, file_name = dataset[idx]
        x = x.unsqueeze(0).to(device)         # (1, T, 34)
        length_tensor = torch.tensor([length]).to(device)
        with torch.no_grad():
            pred = model(x, length_tensor).squeeze(0).cpu().numpy()  # (T,)
        label = y.numpy()  # (T,)

        plot_frame_predictions(pred > 0.5, label, video_name=f"{file_name} ({i+1}/{num_samples})")

# ----- 평가 함수 -----
def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y, lengths, file_names in dataloader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            pred = model(x, lengths)  # (B, T)
            mask = torch.arange(pred.shape[1])[None, :].to(device) < lengths[:, None]
            pred_bin = (pred > 0.5).int()

            all_preds += pred_bin[mask].cpu().tolist()
            all_labels += y[mask].int().cpu().tolist()

            # 첫 배치의 첫 샘플만 시각화
            if len(all_preds) == lengths[0].item():
                plot_frame_predictions(pred[0, :lengths[0]].cpu().numpy(), y[0, :lengths[0]].cpu().numpy(), video_name=file_names[0])
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return acc, f1

# ----- 메인 실행 -----
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dir = "./pose_tensor_npz/pose_tensor_npz/Y/Validation"
    
    print("[INFO] Loading validation data...")
    val_sequences, val_labels, val_lengths, file_names = load_all_npz_in_subfolders(val_dir)
    print(f"[INFO] Loaded {len(val_sequences)} valid sequences.")

    if not val_sequences:
        print("❌ No valid sequences found.")
        exit()

    dataset = FallDataset(val_sequences, val_labels, val_lengths, file_names)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = FrameLSTM()
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)

    acc, f1 = evaluate(model, val_loader)
    print(f"[Validation] Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    # 🔍 추가: 랜덤 3개 시각화
    visualize_random_samples(model, dataset, device)
