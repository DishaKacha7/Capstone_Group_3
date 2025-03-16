import os
import gc
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix, accuracy_score

# Load Preprocessed Data
print("Loading preprocessed train, dev, and test datasets...")

X_train = pd.read_parquet("X_train.parquet")
X_val = pd.read_parquet("X_val.parquet")
X_test = pd.read_parquet("X_test.parquet")

y_train = pd.read_parquet("y_train.parquet")["crop_name_encoded"]
y_val = pd.read_parquet("y_val.parquet")["crop_name_encoded"]
y_test = pd.read_parquet("y_test.parquet")["crop_name_encoded"]

# Load Label Encoder
le = joblib.load('label_encoder.joblib')

# Load or Fit Scaler
scaler_path = "scaler.joblib"
if os.path.exists(scaler_path):
    print("Loading existing scaler...")
    scaler = joblib.load(scaler_path)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
else:
    print("Fitting and saving new scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, scaler_path)

# Train One Model Per Class (OvR)
print("Training models...")
ovr_models = {}
num_classes = len(np.unique(y_train))

# Load models for class 0 and 1 (since they were already trained successfully)
for class_id in [0, 1]:
    model_path = f'ovr_model_class_{class_id}.joblib'
    if os.path.exists(model_path):
        print(f"Loading pre-trained model for class {class_id}...")
        ovr_models[class_id] = joblib.load(model_path)
    else:
        raise ValueError(f"Expected model for class {class_id} but not found!")

# Start training from class 2 onwards, even if models exist
for class_id in range(2, num_classes):
    model_path = f'ovr_model_class_{class_id}.joblib'
    print(f"Training model for class {class_id} (even if it exists)...")

    y_binary = (y_train == class_id).astype(int)
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_binary)

    # Save the trained model
    joblib.dump(model, model_path)
    ovr_models[class_id] = model

    # Free memory
    del model
    gc.collect()

# Generate Meta Features
print("Generating meta features...")
train_meta_features = np.column_stack([model.predict_proba(X_train_scaled)[:, 1] for model in ovr_models.values()])
test_meta_features = np.column_stack([model.predict_proba(X_test_scaled)[:, 1] for model in ovr_models.values()])
dev_meta_features = np.column_stack([model.predict_proba(X_val_scaled)[:, 1] for model in ovr_models.values()])

# Train or Load Meta Model
meta_model_path = "meta_model_class.joblib"
if os.path.exists(meta_model_path):
    print("Loading existing meta-classifier...")
    meta_model = joblib.load(meta_model_path)
else:
    print("Training new meta-classifier...")
    meta_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    meta_model.fit(train_meta_features, y_train)
    joblib.dump(meta_model, meta_model_path)

# Predictions
print("Evaluating on test and dev sets...")
y_test_pred = meta_model.predict(test_meta_features)
y_dev_pred = meta_model.predict(dev_meta_features)


# Function to Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, normalize='true') * 100  # Convert to percentages
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()


# Evaluation Metrics
for y_true, y_pred, name in [(y_test, y_test_pred, "Test Set"), (y_val, y_dev_pred, "Dev Set")]:
    print(f"\n{name} Evaluation:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Cohen's Kappa: {cohen_kappa_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred))
    plot_confusion_matrix(y_true, y_pred, f"Confusion Matrix - {name}")
