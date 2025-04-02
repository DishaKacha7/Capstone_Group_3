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

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ----------------------------------------
# 1. Load and process full dataset
# ----------------------------------------
print("Loading full dataset...")
df = pd.read_parquet("merged_dl_258_259.parquet")  # No sampling, use full data

# ----------------------------------------
# 2. Check for missing values
# ----------------------------------------
print("Any NaNs?", df.isnull().any().any())
print("Any infinite?", np.isinf(df.select_dtypes(include=[np.number])).any().any())
print("NaN count per column:\n", df.isnull().sum())

# ----------------------------------------
# 3. Dynamic numeric column detection & scaling
# ----------------------------------------

# Drop 'May' if it exists
df = df.drop(columns=['May'], errors='ignore')

# Identify numeric columns excluding identifier fields
exclude_cols = {'id', 'point', 'fid', 'crop_id', 'crop_name'}
numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

# Fill NaNs with median before scaling
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Add one-hot encoded 'Type' column (if it exists)
if 'Type' in df.columns:
    df = pd.get_dummies(df, columns=['Type'])

# Combine numeric and one-hot feature columns
one_hot_cols = [col for col in df.columns if col.startswith('Type_')]
feature_columns = numeric_cols + one_hot_cols

# Standardize numeric features
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Extract features
features = df[feature_columns].astype(np.float32)

# Verify no NaNs after processing
print("Any NaNs after fixing?", df[numeric_cols].isnull().any().any())

# ----------------------------------------
# 4. Encode target label
# ----------------------------------------
label_encoder = LabelEncoder()
df['crop_label'] = label_encoder.fit_transform(df['crop_id'])
targets = df['crop_label'].values

# Print class distribution
print("Class distribution:", np.bincount(targets))

# ----------------------------------------
# 5. Train/Val split (no fid overlap)
# ----------------------------------------
print("Splitting train/val without fid overlap...")
unique_fids = df['fid'].unique()
train_fids, val_fids = train_test_split(unique_fids, test_size=0.2, random_state=seed)

train_mask = df['fid'].isin(train_fids)
val_mask = df['fid'].isin(val_fids)

X_train, y_train = features[train_mask], targets[train_mask]
X_val, y_val = features[val_mask], targets[val_mask]

print(f"Train size: {X_train.shape}, Val size: {X_val.shape}")

# ----------------------------------------
# 6. Dataset class
# ----------------------------------------
class CropDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------------------------------
# 7. DataLoaders
# ----------------------------------------
train_dataset = CropDataset(X_train, y_train)
val_dataset = CropDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024)

# ----------------------------------------
# 8. Model
# ----------------------------------------
class CropClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

input_dim = len(feature_columns)
num_classes = len(label_encoder.classes_)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CropClassifier(input_dim, num_classes).to(device)

# ----------------------------------------
# 9. Weighted Loss
# ----------------------------------------
class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ----------------------------------------
# 10. Evaluation function
# ----------------------------------------
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

# ----------------------------------------
# 11. Training loop
# ----------------------------------------
print("Starting training...")
for epoch in range(1, 6):  # Use more epochs later
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

    print(f"Epoch {epoch:02d} | Loss: {running_loss:.4f}")
    evaluate(model, val_loader)

# ----------------------------------------
# 12. Final evaluation: Confusion matrix + Kappa
# ----------------------------------------
print("Final evaluation...")
y_true, y_pred = evaluate(model, val_loader, return_preds=True)

# Convert numerical labels back to crop names
crop_labels = label_encoder.classes_
print("Crop label names:", crop_labels)
# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred, normalize="true")  # Normalize to percentages
cm_percentage = cm * 100  # Convert to percentage

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=crop_labels, yticklabels=crop_labels)

plt.title("Confusion Matrix (Percentage)")
plt.xlabel("Predicted Crop")
plt.ylabel("True Crop")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
