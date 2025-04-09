# ==================== Step 1: Imports ====================
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight
import random
from torchtoolbox.nn import FocalLoss

print("[INFO] Libraries imported.")

# ==================== Step 2: Dataset Loading & Split ====================
print("[INFO] Loading dataset...")
df = pd.read_parquet("merged_dl_258_259.parquet")

print("[INFO] Encoding labels...")
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['crop_name'])

fids = df['fid'].unique()
train_fids, test_fids = train_test_split(fids, test_size=0.2, random_state=42)
train_fids, val_fids = train_test_split(train_fids, test_size=0.1, random_state=42)

train_df = df[df['fid'].isin(train_fids)].reset_index(drop=True)
val_df = df[df['fid'].isin(val_fids)].reset_index(drop=True)
test_df = df[df['fid'].isin(test_fids)].reset_index(drop=True)

print("[INFO] Train/Val/Test split complete.")

# ==================== Step 3: Dataset Class ====================
class CropDataset(Dataset):
    def __init__(self, df, feature_cols):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df['label'].values.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# ==================== Step 4: Feature Columns ====================
feature_cols = [col for col in df.columns if any(b in col for b in ['B2_', 'B6_', 'B11_', 'B12_', 'hue_', 'EVI_'])]

train_dataset = CropDataset(train_df, feature_cols)
val_dataset = CropDataset(val_df, feature_cols)
test_dataset = CropDataset(test_df, feature_cols)

print("[INFO] Dataset classes created.")

# ==================== Step 5: Weighted Sampler ====================
class_counts = np.bincount(train_df['label'])
class_weights_sampler = 1. / class_counts
sample_weights = class_weights_sampler[train_df['label'].values]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

print("[INFO] Weighted sampler initialized.")

# ==================== Step 6: CNN + BiLSTM Model ====================
class CropCNNBiLSTM(nn.Module):
    def __init__(self, input_size, num_classes, conv_filters=64, lstm_hidden=64, kernel_size=5, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(6, conv_filters, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.bilstm = nn.LSTM(input_size=conv_filters, hidden_size=lstm_hidden, num_layers=1,
                              batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * lstm_hidden, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), 6, -1)  # (B, C, T)
        x = self.relu(self.conv1(x))  # (B, F, T)
        x = x.permute(0, 2, 1)        # (B, T, F) for LSTM
        x, _ = self.bilstm(x)         # (B, T, 2*H)
        x = x[:, -1, :]               # take last time step
        x = self.dropout(x)
        return self.fc(x)

print("[INFO] CNN + BiLSTM model class defined.")

# ==================== Step 7: Training & Evaluation ====================
def train(model, optimizer, criterion, dataloader):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def get_logits_and_labels(model, dataloader):
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for X, y in dataloader:
            outputs = model(X)
            logits_list.append(outputs)
            labels_list.extend(y.tolist())
    return torch.cat(logits_list, dim=0), labels_list

print("[INFO] Starting ensemble with CNN + BiLSTM + Focal Loss...")
num_models = 5
logits_ensemble = []

for seed in range(num_models):
    print(f"\n[ENSEMBLE] Training model with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = CropCNNBiLSTM(len(feature_cols), len(label_encoder.classes_))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = FocalLoss(classes=len(label_encoder.classes_), gamma=2.0)

    train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for epoch in range(25):
        loss = train(model, optimizer, criterion, train_loader)
        print(f"[SEED {seed}] Epoch {epoch+1}/25 - Loss: {loss:.4f}")

    logits, test_labels = get_logits_and_labels(model, test_loader)
    torch.save(model.state_dict(), f"model_seed_{seed}_25epochs.pt")
    print(f"[SEED {seed}] Model saved: model_seed_{seed}_25epochs.pt")
    logits_ensemble.append(logits.unsqueeze(0))

# Average logits and compute predictions
print("\n[INFO] Averaging ensemble logits...")
avg_logits = torch.cat(logits_ensemble, dim=0).mean(dim=0)
pred_labels = torch.argmax(avg_logits, dim=1).tolist()

print("[ENSEMBLE + CNN+BiLSTM + FOCAL LOSS] Accuracy:", accuracy_score(test_labels, pred_labels))
print("[ENSEMBLE + CNN+BiLSTM + FOCAL LOSS] F1 Score:", f1_score(test_labels, pred_labels, average='weighted'))
print("[ENSEMBLE + CNN+BiLSTM + FOCAL LOSS] Cohen Kappa:", cohen_kappa_score(test_labels, pred_labels))
print("[ENSEMBLE + CNN+BiLSTM + FOCAL LOSS] Classification Report:\n", classification_report(test_labels, pred_labels, target_names=label_encoder.classes_))
