import pandas as pd
import numpy as np
import torch
import random
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix
from pytorch_tabnet.tab_model import TabNetClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# =================== Step 1: Load Data ===================
df = pd.read_parquet("merged_dl_258_259.parquet")
df = df.drop(columns=['May'], errors='ignore')

# =================== Step 2: Preprocess ===================
exclude_cols = {'id', 'point', 'fid', 'crop_id', 'crop_name'}
numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

if 'Type' in df.columns:
    df = pd.get_dummies(df, columns=['Type'])

one_hot_cols = [col for col in df.columns if col.startswith('Type_')]
feature_columns = numeric_cols + one_hot_cols

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

features = df[feature_columns].astype(np.float32)
label_encoder = LabelEncoder()
df['crop_label'] = label_encoder.fit_transform(df['crop_id'])
targets = df['crop_label'].values

# =================== Step 3: Fid-Wise Split ===================
unique_fids = df['fid'].unique()
trainval_fids, test_fids = train_test_split(unique_fids, test_size=0.2, random_state=42)
train_fids, val_fids = train_test_split(trainval_fids, test_size=0.2, random_state=42)

train_mask = df['fid'].isin(train_fids)
val_mask = df['fid'].isin(val_fids)
test_mask = df['fid'].isin(test_fids)

X_train, y_train = features[train_mask].values, targets[train_mask]
X_val, y_val = features[val_mask].values, targets[val_mask]
X_test, y_test = features[test_mask].values, targets[test_mask]

# =================== Step 4: Ensemble of TabTransformer ===================
n_models = 5
seeds = [42, 101, 202, 303, 404]
val_preds_all, test_preds_all = [], []
model_dir = "saved_models_tabnet"
os.makedirs(model_dir, exist_ok=True)

for seed in seeds:
    model_path = os.path.join(model_dir, f"tabnet_seed_{seed}.zip")
    model = TabNetClassifier(
        n_d=64, n_a=64, n_steps=5,
        gamma=1.5, n_independent=2, n_shared=2,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-3),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=seed,
        verbose=1
    )

    if os.path.exists(model_path):
        print(f"🔁 Loading saved model for seed {seed}...")
        model.load_model(model_path)
    else:
        print(f"🚀 Training new model for seed {seed}...")
        model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=["accuracy"],
            max_epochs=100,
            patience=30,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        model.save_model(model_path)
        print(f"Saved model to {model_path}")

    val_preds_all.append(model.predict_proba(X_val))
    test_preds_all.append(model.predict_proba(X_test))

# =================== Step 5: Aggregated Predictions ===================
val_pred_mean = np.mean(val_preds_all, axis=0)
test_pred_mean = np.mean(test_preds_all, axis=0)

y_val_pred = np.argmax(val_pred_mean, axis=1)
y_test_pred = np.argmax(test_pred_mean, axis=1)

# =================== Step 6: Field-Level Aggregation ===================
def aggregate_field_preds(df_subset, y_preds, label_col='crop_label'):
    pred_df = pd.DataFrame({
        'fid': df_subset['fid'].values,
        'pred_label': y_preds,
        'true_label': df_subset[label_col].values
    })
    field_pred = pred_df.groupby('fid')['pred_label'].agg(lambda x: Counter(x).most_common(1)[0][0])
    field_true = pred_df.groupby('fid')['true_label'].agg(lambda x: Counter(x).most_common(1)[0][0])
    return field_true, field_pred

train_field_true, train_field_pred = aggregate_field_preds(df[train_mask], targets[train_mask])
val_field_true, val_field_pred = aggregate_field_preds(df[val_mask], y_val_pred)
test_field_true, test_field_pred = aggregate_field_preds(df[test_mask], y_test_pred)

# =================== Step 7: Evaluation Function ===================
def evaluate_field_level(true_labels, pred_labels, title="Confusion Matrix"):
    print("Accuracy:", accuracy_score(true_labels, pred_labels))
    print("F1 Score:", f1_score(true_labels, pred_labels, average='weighted'))
    print("Cohen Kappa:", cohen_kappa_score(true_labels, pred_labels))
    print("Classification Report:")
    target_names = [str(cls) for cls in label_encoder.classes_]
    print(classification_report(true_labels, pred_labels, target_names=target_names))

    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# =================== Step 8: Results ===================
print("========= [Train Field-Level Evaluation] =========")
evaluate_field_level(train_field_true, train_field_pred, title="Train Confusion Matrix")

print("========= [Validation Field-Level Evaluation] =========")
evaluate_field_level(val_field_true, val_field_pred, title="Validation Confusion Matrix")

print("========= [Test Field-Level Evaluation] =========")
evaluate_field_level(test_field_true, test_field_pred, title="Test Confusion Matrix")
