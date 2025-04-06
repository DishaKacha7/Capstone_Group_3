# ==================== Step 1: Imports ====================
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, cohen_kappa_score
import optuna

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
class_weights = 1. / class_counts
sample_weights = class_weights[train_df['label'].values]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

print("[INFO] Weighted sampler initialized.")

# ==================== Step 6: Model Definition ====================
class CropCNN1D(nn.Module):
    def __init__(self, input_size, num_classes, conv_filters=64, kernel_size=7, dropout=0.4939626568904243):
        super().__init__()
        self.conv1 = nn.Conv1d(6, conv_filters, kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(conv_filters, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), 6, -1)
        x = self.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

print("[INFO] CNN model class defined.")

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

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in dataloader:
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())
    return all_labels, all_preds

print("[INFO] Training final model with best hyperparameters...")
model = CropCNN1D(len(feature_cols), len(label_encoder.classes_))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001162338603462791)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

for epoch in range(50):
    loss = train(model, optimizer, criterion, train_loader)
    print(f"[FINAL MODEL] Epoch {epoch+1}/50 - Loss: {loss:.4f}")

# Evaluate on Validation
val_labels, val_preds = evaluate_model(model, val_loader)
print("[VALIDATION] Accuracy:", accuracy_score(val_labels, val_preds))
print("[VALIDATION] F1 Score:", f1_score(val_labels, val_preds, average='weighted'))
print("[VALIDATION] Cohen Kappa:", cohen_kappa_score(val_labels, val_preds))
print("[VALIDATION] Classification Report:\n", classification_report(val_labels, val_preds, target_names=label_encoder.classes_))

# Evaluate on Test
test_labels, test_preds = evaluate_model(model, test_loader)
print("[TEST] Accuracy:", accuracy_score(test_labels, test_preds))
print("[TEST] F1 Score:", f1_score(test_labels, test_preds, average='weighted'))
print("[TEST] Cohen Kappa:", cohen_kappa_score(test_labels, test_preds))
print("[TEST] Classification Report:\n", classification_report(test_labels, test_preds, target_names=label_encoder.classes_))
