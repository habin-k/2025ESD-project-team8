import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# 1) Data loading for Training / Validation
def load_npz_frames(root_dir, split):
    seqs, labs, lengths = [], [], []
    for label_dir in ['Y','N']:
        base = os.path.join(root_dir, label_dir, split)
        for subdir, _, files in os.walk(base):
            for fn in files:
                if not fn.endswith('.npz'):
                    continue
                data = np.load(os.path.join(subdir, fn), allow_pickle=True)
                if 'pose' in data.files and 'label' in data.files:
                    p = data['pose']   # (20,17,2)
                    l = data['label']  # (20,)
                    if p.ndim==3 and p.shape[1:]==(17,2) and l.ndim==1 and l.shape[0]==p.shape[0]:
                        T = p.shape[0]
                        p = p.reshape(T, -1)  # (T,34)
                        # (0,0) frame correction
                        mask0 = (p==0).all(axis=1)
                        for t in range(1, T):
                            if mask0[t]:
                                p[t] = p[t-1]
                        seqs.append(torch.tensor(p, dtype=torch.float32))
                        labs.append(torch.tensor(l, dtype=torch.float32))
                        lengths.append(T)
    return seqs, labs, lengths

# Paths
root = "/content/drive/MyDrive/pose_tensor_npz"
train_seqs, train_labs, train_lens = load_npz_frames(root, 'Training')
val_seqs,   val_labs,   val_lens   = load_npz_frames(root, 'Validation')

# 2) Dataset & DataLoader
class FallFrameDataset(Dataset):
    def __init__(self, seqs, labs, lengths):
        self.seqs, self.labs, self.lengths = seqs, labs, lengths
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, i):
        return self.seqs[i], self.labs[i], self.lengths[i]

def collate_fn(batch):
    seqs, labs, lens = zip(*batch)
    X = pad_sequence(seqs, batch_first=True)       # (B, T_max, 34)
    Y = pad_sequence(labs, batch_first=True).float()  # (B, T_max), float
    L = torch.tensor(lens, dtype=torch.long)       # (B,)
    return X, Y, L

train_loader = DataLoader(
    FallFrameDataset(train_seqs, train_labs, train_lens),
    batch_size=16, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    FallFrameDataset(val_seqs, val_labs, val_lens),
    batch_size=16, shuffle=False, collate_fn=collate_fn
)

# 3) Model: 2-layer BiLSTM + dropout
class FrameLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size*2, 1)
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out_unpad, _ = pad_packed_sequence(out_packed, batch_first=True)
        return self.fc(out_unpad).squeeze(-1)  # (B, T_max)

model = FrameLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# pos_weight for imbalance
all_train = torch.cat(train_labs)
pos = all_train.sum()
neg = all_train.numel() - pos
criterion = nn.BCEWithLogitsLoss(pos_weight=(neg/pos))

# Scheduler & EarlyStopping
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)
best_val_f1 = 0
no_improve = 0
patience = 5


THRESHOLD     = 0.75    # Frame-level 분류 임계값

# 4) Training loop
train_losses, train_f1s, val_f1s = [], [], []

for epoch in range(1, 51):
    # --- Training ---
    model.train()
    epoch_loss = 0.0
    t_preds, t_trues = [], []
    for X, Y, L in train_loader:
        logits = model(X, L)  # (B, T_max)

        # 1) Loss 계산 (패딩 제외 평균)
        mask = (torch.arange(logits.size(1))[None] < L[:, None]).float()
        loss = (criterion(logits, Y) * mask).sum() / mask.sum()
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        p = (torch.sigmoid(logits) > THRESHOLD)[mask.bool()]
        t_preds += p.int().tolist()
        t_trues += Y[mask.bool()].int().tolist()

    # 에포크당 평균 손실 & F1 계산
    avg_loss = epoch_loss / len(train_loader)
    tr_f1 = f1_score(t_trues, t_preds)
    train_losses.append(avg_loss)
    train_f1s.append(tr_f1)


    # Validation + event metrics
    model.eval()
    preds, trues = [], []
    total_events = detected_events = 0
    delays = []
    with torch.no_grad():
        for X, Y, L in val_loader:
            logits = model(X, L)
            probs = torch.sigmoid(logits)
            B = logits.size(0)
            for b in range(B):
                T = L[b].item()
                seq_true = Y[b, :T].int().tolist()
                seq_pred = (probs[b, :T] > THRESHOLD).tolist()
                # frame-level
                preds += list(map(int, seq_pred))
                trues += seq_true
                # event-level
                if any(seq_true):
                    total_events += 1
                    first_true = seq_true.index(1)
                    if any(seq_pred):
                        detected_events += 1
                        first_pred = seq_pred.index(True)
                        delays.append(max(0, first_pred - first_true))
    val_f1 = f1_score(trues, preds)
    val_f1s.append(val_f1)
    evt_rec = detected_events / total_events if total_events else 0
    avg_delay = sum(delays) / len(delays) if delays else 0.0


    # 로그 출력: Loss, Train F1, Val F1, Event Recall, Delay
    print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | "
          f"Train F1: {tr_f1:.3f} | Val F1: {val_f1s[-1]:.3f} | "
          f"Evt Rec: {evt_rec:.3f} | Delay: {avg_delay:.2f}")


    scheduler.step(val_f1)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'best_model.pth')
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping.")
            break

# 5) Plot progress
epochs = list(range(1, len(train_f1s)+1))
plt.figure()
plt.plot(epochs, train_losses, label='loss')
plt.plot(epochs, train_f1s, label='Train F1')
plt.plot(epochs, val_f1s,   label='Val F1')
plt.axhline(0.80, color='gray', linestyle='--', label='Target F1=0.80')
plt.xlabel('Epoch'); plt.ylabel('Frame F1'); plt.legend(); plt.show()

