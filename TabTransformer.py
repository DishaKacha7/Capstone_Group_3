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
import random

# Set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Load data
df = pd.read_parquet("merged_dl_258_259.parquet")

# Drop and define features
drop_cols = ['id', 'fid', 'point', 'crop_name', 'crop_id']
feature_cols = [col for col in df.columns if col not in drop_cols + ['crop_label']]
label_encoder = LabelEncoder()
df['crop_label'] = label_encoder.fit_transform(df['crop_id'])

# Normalize & fill missing
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])
X_all = df[feature_cols].values.astype(np.float32)
targets = df['crop_label'].values

# Train/val split
unique_fids = df['fid'].unique()
train_fids, val_fids = train_test_split(unique_fids, test_size=0.2, random_state=seed)
train_mask = df['fid'].isin(train_fids)
val_mask = df['fid'].isin(val_fids)

X_train, y_train = X_all[train_mask], targets[train_mask]
X_val, y_val = X_all[val_mask], targets[val_mask]

# Dataset
class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TabDataset(X_train, y_train)
val_dataset = TabDataset(X_val, y_val)

# Weighted sampler
class_sample_counts = np.bincount(y_train)
weights = 1. / class_sample_counts
sample_weights = weights[y_train]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=256)

# TabTransformer model
class TabTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, depth=2, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # (batch, seq_len=1, dim)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.classifier(x)

# Model training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(label_encoder.classes_)
model = TabTransformer(input_dim=X_train.shape[1], num_classes=num_classes).to(device)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

# Evaluation
def evaluate(model, loader, return_preds=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    print(f"✅ Accuracy: {acc:.4f} | Cohen Kappa: {kappa:.4f}")
    if return_preds:
        return all_labels, all_preds
    return acc

# Training loop
print("Training TabTransformer...")
best_val_loss = float('inf')
patience = 5
no_improve = 0

for epoch in range(1, 20):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

    scheduler.step(val_loss)
    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    evaluate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping.")
            break

# Final Evaluation
y_true, y_pred = evaluate(model, val_loader, return_preds=True)
decoded_labels = df[['crop_label', 'crop_name']].drop_duplicates().sort_values('crop_label')
crop_names = decoded_labels['crop_name'].tolist()

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=crop_names, digits=3))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, normalize="true") * 100
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", xticklabels=crop_names, yticklabels=crop_names)
plt.title("Confusion Matrix (Normalized %)")
plt.xlabel("Predicted Crop")
plt.ylabel("True Crop")
plt.tight_layout()
plt.show()
