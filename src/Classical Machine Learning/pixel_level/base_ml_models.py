# ===================== IMPORTS =====================
import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

# ===================== DATA PREPROCESSING =====================
def prepare_data(df, label_encoder, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ['id', 'point', 'fid', 'crop_id', 'SHAPE_AREA', 'SHAPE_LEN']
    feature_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in feature_cols if col not in exclude_cols and not df[col].isna().any()]
    X = df[feature_cols]
    y_encoded = label_encoder.transform(df['crop_name'])
    return X, y_encoded, feature_cols

# ===================== MODEL TRAIN & PREDICT =====================
def train_predict_model(model, X_train_scaled, y_train, X_test_scaled):
    print("Training model...")
    model.fit(X_train_scaled, y_train)
    print("Model trained. Predicting...")
    return model.predict(X_test_scaled)

# ===================== FIELD-WISE AGGREGATION AND EVALUATION =====================
def evaluate_field_level(test_df, y_test, y_pred, label_encoder, model_name, save_path):
    print(f"Aggregating pixel-wise predictions into field-wise predictions for {model_name}...")
    pixel_preds = pd.DataFrame({
        'fid': test_df['fid'].values,
        'true_label': label_encoder.inverse_transform(y_test),
        'predicted_label': label_encoder.inverse_transform(y_pred)
    })

    field_predictions = (
        pixel_preds.groupby('fid')
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
    )

    true_labels_field = field_predictions['true_label']
    predicted_labels_field = field_predictions['predicted_label']

    acc = accuracy_score(true_labels_field, predicted_labels_field)
    f1_macro = f1_score(true_labels_field, predicted_labels_field, average='macro')
    f1_weighted = f1_score(true_labels_field, predicted_labels_field, average='weighted')
    kappa = cohen_kappa_score(true_labels_field, predicted_labels_field)

    print(f"Field-level Accuracy: {acc:.4f}")
    print(f"Field-level F1 Macro: {f1_macro:.4f}")
    print(f"Field-level F1 Weighted: {f1_weighted:.4f}")
    print(f"Field-level Cohen's Kappa: {kappa:.4f}")

    field_predictions.to_csv(os.path.join(save_path, f'{model_name.lower().replace(" ", "_")}_field_predictions.csv'), index=False)

# ===================== MAIN =====================
if __name__ == "__main__":

    # Create base directory
    base_dir = 'ml_base'
    os.makedirs(base_dir, exist_ok=True)

    # Load Data
    print("Loading dataset...")
    file_path = "./Data/final_data.parquet"
    data = pd.read_parquet(file_path)

    # Split by Field
    print("Splitting data by field IDs...")
    fids = data['fid'].unique()
    train_fids, test_fids = train_test_split(fids, test_size=0.2, random_state=42)
    train_df = data[data['fid'].isin(train_fids)].copy()
    test_df = data[data['fid'].isin(test_fids)].copy()

    # Prepare Data
    print("Preparing data...")
    label_encoder = LabelEncoder()
    label_encoder.fit(data['crop_name'])
    X_train, y_train, feature_cols = prepare_data(train_df, label_encoder)
    X_test, y_test, _ = prepare_data(test_df, label_encoder)

    # Scale
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=100, n_jobs=-1),
        'Random Forest': RandomForestClassifier(n_estimators=20, n_jobs=-1),
        'LightGBM': lgb.LGBMClassifier(n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss', n_jobs=-1)
    }

    # Save Directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(base_dir, timestamp)
    os.makedirs(save_path, exist_ok=True)

    print(f"Saving preprocessing artifacts to {save_path}...")
    joblib.dump(scaler, os.path.join(save_path, 'scaler.joblib'))
    joblib.dump(label_encoder, os.path.join(save_path, 'label_encoder.joblib'))
    joblib.dump(feature_cols, os.path.join(save_path, 'feature_columns.joblib'))

    # Train, Predict, Evaluate and Save CSVs
    for model_name, model in models.items():
        print(f"\nTraining and evaluating {model_name}...")
        y_pred = train_predict_model(model, X_train_scaled, y_train, X_test_scaled)

        # Save model
        print(f"Saving model {model_name}...")
        joblib.dump(model, os.path.join(save_path, f'{model_name.lower().replace(" ", "_")}.joblib'))

        # Save field-level evaluations
        evaluate_field_level(test_df, y_test, y_pred, label_encoder, model_name, save_path)

    print(f"\nAll models and outputs saved in: {save_path}")
