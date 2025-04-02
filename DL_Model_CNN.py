import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Set random seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Load dataset
df = pd.read_parquet("merged_dl_258_259.parquet")

# Drop unwanted columns and prepare features
drop_cols = ['id', 'point', 'fid', 'crop_id', 'crop_name']
feature_cols = [col for col in df.columns if col not in drop_cols + ['crop_label']]

# Encode target labels
label_encoder = LabelEncoder()
df['crop_label'] = label_encoder.fit_transform(df['crop_id'])
targets = df['crop_label'].values

# Fill missing values and normalize
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Feature matrix
X_all = df[feature_cols].values.astype(np.float32)

# Train/validation split by 'fid'
unique_fids = df['fid'].unique()
train_fids, val_fids = train_test_split(unique_fids, test_size=0.2, random_state=seed)
train_mask = df['fid'].isin(train_fids)
val_mask = df['fid'].isin(val_fids)

X_train, y_train = X_all[train_mask], targets[train_mask]
X_val, y_val = X_all[val_mask], targets[val_mask]

# Dataset class
class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Dataloaders
train_dataset = SimpleDataset(X_train, y_train)
val_dataset = SimpleDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)

# MLP Model
class MLPModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

num_classes = len(label_encoder.classes_)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPModel(input_dim=X_train.shape[1], num_classes=num_classes).to(device)

# Weighted loss for class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

# Evaluation function
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
    print(f"\u2705 Accuracy: {acc:.4f} | Cohen Kappa: {kappa:.4f}")

    if return_preds:
        return all_labels, all_preds
    return acc

# Training loop
print("Starting training...")
best_val_loss = float('inf')
patience = 5
no_improve = 0
for epoch in range(1, 50):
    model.train()
    running_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

    scheduler.step(val_loss)

    print(f"Epoch {epoch:02d} | Train Loss: {running_loss:.4f} | Val Loss: {val_loss:.4f}")
    evaluate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping triggered.")
            break

# Final evaluation and confusion matrix
y_true, y_pred = evaluate(model, val_loader, return_preds=True)
decoded_labels = df[['crop_label', 'crop_name']].drop_duplicates().sort_values('crop_label')
crop_names = decoded_labels['crop_name'].tolist()

print("Crop names:", crop_names)

cm = confusion_matrix(y_true, y_pred, normalize="true") * 100
plt.figure(figsize=(12, 10))
ax = sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                 xticklabels=crop_names, yticklabels=crop_names)
ax.set_xticklabels(crop_names, rotation=45, ha="right")
ax.set_yticklabels(crop_names, rotation=0)
plt.title("Confusion Matrix (Percentage)")
plt.xlabel("Predicted Crop")
plt.ylabel("True Crop")
plt.tight_layout()
plt.show()
