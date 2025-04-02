# Hybrid LSTM + MLP Model with Early Stopping on Cohen's Kappa

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Set random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Load data
df = pd.read_parquet("merged_dl_258_259.parquet")

# Encode target
df['crop_label'] = LabelEncoder().fit_transform(df['crop_id'])

# Split time-series vs shape features
shape_features = ['SHAPE_AREA', 'SHAPE_LEN']
month_pattern = re.compile(r'_(2017)_(0[1-9]|1[0-2])')
monthly_features = [col for col in df.columns if month_pattern.search(col)]
monthly_df = df[monthly_features]
static_df = df[shape_features]

# Remove May (05) and June (06)
monthly_df = monthly_df[[col for col in monthly_df.columns if '_2017_05' not in col and '_2017_06' not in col]]

# Train-val split
train_fids, val_fids = train_test_split(df['fid'].unique(), test_size=0.2, random_state=seed)
train_mask = df['fid'].isin(train_fids)
val_mask = df['fid'].isin(val_fids)

# Fill NA and scale separately to avoid leakage
monthly_train = monthly_df[train_mask].copy()
monthly_val = monthly_df[val_mask].copy()
static_train = static_df[train_mask].copy()
static_val = static_df[val_mask].copy()

monthly_scaler = StandardScaler()
static_scaler = StandardScaler()

monthly_train = pd.DataFrame(monthly_scaler.fit_transform(monthly_train), columns=monthly_train.columns)
monthly_val = pd.DataFrame(monthly_scaler.transform(monthly_val), columns=monthly_val.columns)

static_train = pd.DataFrame(static_scaler.fit_transform(static_train), columns=static_train.columns)
static_val = pd.DataFrame(static_scaler.transform(static_val), columns=static_val.columns)

# Prepare monthly sequences
months = sorted(set([m[-2:] for m in monthly_train.columns]))
time_feat_map = {m: [col for col in monthly_train.columns if col.endswith(f'_{m}')] for m in months}

monthly_tensor_train = np.stack([monthly_train[time_feat_map[m]].values for m in months], axis=1)
monthly_tensor_val = np.stack([monthly_val[time_feat_map[m]].values for m in months], axis=1)

static_tensor_train = static_train.values
static_tensor_val = static_val.values

labels_train = df[train_mask]['crop_label'].values
labels_val = df[val_mask]['crop_label'].values

# Dataset
class CropDataset(Dataset):
    def __init__(self, X_seq, X_static, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_static = torch.tensor(X_static, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_static[idx], self.y[idx]

train_dataset = CropDataset(monthly_tensor_train, static_tensor_train, labels_train)
val_dataset = CropDataset(monthly_tensor_val, static_tensor_val, labels_val)

# Weighted sampler
class_counts = np.bincount(labels_train)
weights = 1. / class_counts
sample_weights = weights[labels_train]
sampler = WeightedRandomSampler(sample_weights, len(labels_train), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=256)

# Model
class LSTM_MLP_Hybrid(nn.Module):
    def __init__(self, input_seq_dim, static_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_seq_dim, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.static_branch = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_seq, x_static):
        lstm_out, _ = self.lstm(x_seq)
        lstm_feat = lstm_out[:, -1, :]
        static_feat = self.static_branch(x_static)
        combined = torch.cat([lstm_feat, static_feat], dim=1)
        return self.classifier(combined)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(np.unique(labels_train))
model = LSTM_MLP_Hybrid(monthly_tensor_train.shape[2], static_tensor_train.shape[1], num_classes).to(device)

class_weights = compute_class_weight('balanced', classes=np.unique(labels_train), y=labels_train)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

# Evaluation
def evaluate(model, loader, return_preds=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_seq, x_static, y in loader:
            x_seq, x_static = x_seq.to(device), x_static.to(device)
            logits = model(x_seq, x_static)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    print(f"\u2705 Accuracy: {acc:.4f} | Cohen Kappa: {kappa:.4f}")
    if return_preds:
        return all_labels, all_preds
    return kappa

# Training loop with early stopping on Kappa
print("Training Hybrid LSTM + MLP model...")
best_kappa = -1
patience = 5
no_improve = 0

for epoch in range(1, 30):
    model.train()
    train_loss = 0
    for x_seq, x_static, y in train_loader:
        x_seq, x_static, y = x_seq.to(device), x_static.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x_seq, x_static)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for x_seq, x_static, y in val_loader:
            x_seq, x_static, y = x_seq.to(device), x_static.to(device), y.to(device)
            logits = model(x_seq, x_static)
            loss = criterion(logits, y)
            val_loss += loss.item()

    scheduler.step(val_loss)
    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    kappa = evaluate(model, val_loader)

    if kappa > best_kappa:
        best_kappa = kappa
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping.")
            break

# Final evaluation
y_true, y_pred = evaluate(model, val_loader, return_preds=True)
decoded_labels = df[['crop_label', 'crop_name']].drop_duplicates().sort_values('crop_label')
crop_names = decoded_labels['crop_name'].tolist()

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=crop_names, digits=3))

cm = confusion_matrix(y_true, y_pred, normalize="true") * 100
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", xticklabels=crop_names, yticklabels=crop_names)
plt.title("Confusion Matrix (Normalized %)")
plt.xlabel("Predicted Crop")
plt.ylabel("True Crop")
plt.tight_layout()
plt.show()
