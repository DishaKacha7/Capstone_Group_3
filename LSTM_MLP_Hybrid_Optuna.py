# Hybrid LSTM + MLP Model with Optuna Hyperparameter Optimization

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
import optuna

# Set random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Load and preprocess data
df = pd.read_parquet("merged_dl_258_259.parquet")
df['crop_label'] = LabelEncoder().fit_transform(df['crop_id'])

shape_features = ['SHAPE_AREA', 'SHAPE_LEN']
month_pattern = re.compile(r'_(2017)_(0[1-9]|1[0-2])')
monthly_features = [col for col in df.columns if month_pattern.search(col)]
monthly_df = df[monthly_features]
static_df = df[shape_features]
monthly_df = monthly_df[[col for col in monthly_df.columns if '_2017_05' not in col and '_2017_06' not in col]]

train_fids, val_fids = train_test_split(df['fid'].unique(), test_size=0.2, random_state=seed)
train_mask = df['fid'].isin(train_fids)
val_mask = df['fid'].isin(val_fids)

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

months = sorted(set([m[-2:] for m in monthly_train.columns]))
time_feat_map = {m: [col for col in monthly_train.columns if col.endswith(f'_{m}')] for m in months}

monthly_tensor_train = np.stack([monthly_train[time_feat_map[m]].values for m in months], axis=1)
monthly_tensor_val = np.stack([monthly_val[time_feat_map[m]].values for m in months], axis=1)

static_tensor_train = static_train.values
static_tensor_val = static_val.values

labels_train = df[train_mask]['crop_label'].values
labels_val = df[val_mask]['crop_label'].values

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

# Model Definition
def build_model(input_seq_dim, static_dim, num_classes, hidden_size, num_layers, dropout):
    class LSTM_MLP_Hybrid(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_seq_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
            self.static_branch = nn.Sequential(
                nn.Linear(static_dim, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32)
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size * 2 + 32, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes)
            )

        def forward(self, x_seq, x_static):
            lstm_out, _ = self.lstm(x_seq)
            lstm_feat = lstm_out[:, -1, :]
            static_feat = self.static_branch(x_static)
            combined = torch.cat([lstm_feat, static_feat], dim=1)
            return self.classifier(combined)

    return LSTM_MLP_Hybrid()

# Optuna objective
def objective(trial):
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

    model = build_model(
        input_seq_dim=monthly_tensor_train.shape[2],
        static_dim=static_tensor_train.shape[1],
        num_classes=len(np.unique(labels_train)),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to("cuda")

    class_weights = compute_class_weight('balanced', classes=np.unique(labels_train), y=labels_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to("cuda"))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    sampler = WeightedRandomSampler(1. / np.bincount(labels_train)[labels_train], len(labels_train), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_kappa = -1
    patience = 5
    no_improve = 0

    for epoch in range(15):
        model.train()
        for x_seq, x_static, y in train_loader:
            x_seq, x_static, y = x_seq.cuda(), x_static.cuda(), y.cuda()
            optimizer.zero_grad()
            logits = model(x_seq, x_static)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_seq, x_static, y in val_loader:
                x_seq, x_static = x_seq.cuda(), x_static.cuda()
                logits = model(x_seq, x_static)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())

        kappa = cohen_kappa_score(all_labels, all_preds)
        if kappa > best_kappa:
            best_kappa = kappa
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_kappa

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("Best trial:")
print(study.best_trial)
print("Best params:")
print(study.best_params)

# Final Evaluation Using Best Parameters
best_params = study.best_params
model = build_model(
    input_seq_dim=monthly_tensor_train.shape[2],
    static_dim=static_tensor_train.shape[1],
    num_classes=len(np.unique(labels_train)),
    hidden_size=best_params["hidden_size"],
    num_layers=best_params["num_layers"],
    dropout=best_params["dropout"]
).to("cuda")

criterion = nn.CrossEntropyLoss(weight=torch.tensor(
    compute_class_weight('balanced', classes=np.unique(labels_train), y=labels_train),
    dtype=torch.float32
).to("cuda"))

optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
sampler = WeightedRandomSampler(1. / np.bincount(labels_train)[labels_train], len(labels_train), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"])

# Train best model briefly
for epoch in range(5):
    model.train()
    for x_seq, x_static, y in train_loader:
        x_seq, x_static, y = x_seq.cuda(), x_static.cuda(), y.cuda()
        optimizer.zero_grad()
        logits = model(x_seq, x_static)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

# Final Predictions
y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for x_seq, x_static, y in val_loader:
        x_seq, x_static = x_seq.cuda(), x_static.cuda()
        logits = model(x_seq, x_static)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_true.extend(y.numpy())
        y_pred.extend(preds)

# Classification Report
crop_names = df[['crop_label', 'crop_name']].drop_duplicates().sort_values('crop_label')['crop_name'].tolist()
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