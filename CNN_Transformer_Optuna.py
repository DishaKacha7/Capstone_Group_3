# CNN + Transformer Hybrid Model with Improved Optuna Tuning and Focal Loss

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import re
import torch.nn.functional as F

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

months = sorted(set([m[-2:] for m in monthly_df.columns]))
time_feat_map = {m: [col for col in monthly_df.columns if col.endswith(f'_{m}')] for m in months}

# Define dataset
class CropDataset(Dataset):
    def __init__(self, X_seq, X_static, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_static = torch.tensor(X_static, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_static[idx], self.y[idx]

# Focal loss for imbalanced classification
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        logpt = -F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(logpt)
        loss = ((1 - pt) ** self.gamma * logpt).mean()
        return loss

# Model
class CNNTransformerHybrid(nn.Module):
    def __init__(self, seq_input_dim, static_dim, num_classes, cnn_out_channels, transformer_dim,
                 num_heads, dropout, num_layers, static_hidden_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(seq_input_dim, cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels)
        )
        self.pos_encoder = nn.Parameter(torch.randn(1, 10, transformer_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(cnn_out_channels, transformer_dim)
        self.static_branch = nn.Sequential(
            nn.Linear(static_dim, static_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(static_hidden_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim + static_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_seq, x_static):
        x_seq = x_seq.permute(0, 2, 1)
        cnn_out = self.cnn(x_seq)
        cnn_out = cnn_out.permute(0, 2, 1)
        transformer_in = self.proj(cnn_out) + self.pos_encoder
        transformer_out = self.transformer(transformer_in)
        transformer_feat = transformer_out.mean(dim=1)
        static_feat = self.static_branch(x_static)
        combined = torch.cat([transformer_feat, static_feat], dim=1)
        return self.classifier(combined)

# Objective function with Optuna

def objective(trial):
    cnn_out_channels = trial.suggest_categorical("cnn_out_channels", [32, 64, 128])
    transformer_dim = trial.suggest_categorical("transformer_dim", [64, 128])
    num_heads = trial.suggest_categorical("num_heads", [2, 4])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    static_hidden_dim = trial.suggest_categorical("static_hidden_dim", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256])

    train_fids, val_fids = train_test_split(df['fid'].unique(), test_size=0.2, random_state=seed)
    train_mask = df['fid'].isin(train_fids)
    val_mask = df['fid'].isin(val_fids)

    X_seq_train = np.stack([monthly_df[train_mask][time_feat_map[m]].values for m in months], axis=1)
    X_seq_val = np.stack([monthly_df[val_mask][time_feat_map[m]].values for m in months], axis=1)

    X_static_train = static_df[train_mask].values
    X_static_val = static_df[val_mask].values

    y_train = df[train_mask]['crop_label'].values
    y_val = df[val_mask]['crop_label'].values

    monthly_scaler = StandardScaler()
    static_scaler = StandardScaler()
    X_seq_train = monthly_scaler.fit_transform(X_seq_train.reshape(-1, X_seq_train.shape[-1])).reshape(X_seq_train.shape)
    X_seq_val = monthly_scaler.transform(X_seq_val.reshape(-1, X_seq_val.shape[-1])).reshape(X_seq_val.shape)
    X_static_train = static_scaler.fit_transform(X_static_train)
    X_static_val = static_scaler.transform(X_static_val)

    train_dataset = CropDataset(X_seq_train, X_static_train, y_train)
    val_dataset = CropDataset(X_seq_val, X_static_val, y_val)

    model = CNNTransformerHybrid(
        seq_input_dim=X_seq_train.shape[2],
        static_dim=X_static_train.shape[1],
        num_classes=len(np.unique(y_train)),
        cnn_out_channels=cnn_out_channels,
        transformer_dim=transformer_dim,
        num_heads=num_heads,
        dropout=dropout,
        num_layers=num_layers,
        static_hidden_dim=static_hidden_dim
    ).to("cuda")

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    criterion = FocalLoss(gamma=2.0, weight=torch.tensor(class_weights, dtype=torch.float32).to("cuda"))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    sampler = WeightedRandomSampler(1. / np.bincount(y_train)[y_train], len(y_train), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_kappa = -1
    for epoch in range(20):
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
        best_kappa = max(best_kappa, kappa)

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, digits=3))
    cm = confusion_matrix(all_labels, all_preds, normalize="true") * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")
    plt.title("Confusion Matrix (Normalized %)")
    plt.xlabel("Predicted Crop")
    plt.ylabel("True Crop")
    plt.tight_layout()
    plt.show()

    return best_kappa

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best trial:", study.best_trial.value)
print("Best params:", study.best_params)
