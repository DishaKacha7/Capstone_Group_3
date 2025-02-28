import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import joblib

print("Loading data...")
file_path = "final_data.parquet"
data = pd.read_parquet(file_path, engine="pyarrow").sample(frac=1, random_state=42)

print("Encoding labels...")
le = LabelEncoder()
data['crop_name_encoded'] = le.fit_transform(data['crop_name'])

print("Selecting features...")
exclude_cols = ['id', 'point', 'fid', 'crop_id', 'SHAPE_AREA', 'SHAPE_LEN', 'crop_name']
feature_cols = [col for col in data.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
X = data[feature_cols]
y = data['crop_name_encoded']

print("Handling missing values...")
X = X.dropna(axis=1, how='all')
feature_cols = X.columns.tolist()
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

# Feature Selection: Remove Low Variance Features
print("Removing low variance features...")
selector = VarianceThreshold(threshold=0.01)  # Remove near-constant features
X = pd.DataFrame(selector.fit_transform(X), columns=[feature_cols[i] for i in range(len(feature_cols)) if selector.variances_[i] > 0.01])

# Feature Selection: Remove Highly Correlated Features
print("Removing highly correlated features...")
corr_matrix = X.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.9)]
X.drop(columns=to_drop, inplace=True)

# Feature Selection: Top 10 Most Important Features from Random Forest
print("Selecting top 10 important features...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(X, y)
feature_importances = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10)
top_features = feature_importances.index.tolist()
X = X[top_features]

# Train-Test-Validation Split
print("Splitting data into train, validation, and test sets...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Apply SMOTE only to the Training Set
print("Applying SMOTE for class balancing...")
smote = SMOTE(sampling_strategy='not majority', random_state=42)  # Balance only minority classes
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:", Counter(y_train_resampled))

# Scale Data
print("Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Base Models with Early Stopping
print("Defining base models...")
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)),
    ('lgbm', lgb.LGBMClassifier(n_jobs=-1, early_stopping_rounds=10)),
    ('xgb', xgb.XGBClassifier(eval_metric='logloss', n_jobs=-1, early_stopping_rounds=10))
]

print("Training base models...")
for name, model in base_models:
    print(f"Training {name}...")
    if name in ['lgbm', 'xgb']:  # Use validation set for early stopping
        model.fit(X_train_scaled, y_train_resampled, eval_set=[(X_val_scaled, y_val)], verbose=False)
    else:
        model.fit(X_train_scaled, y_train_resampled)

# Train Final Model
print("Training final model...")
final_model = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
final_model.fit(X_train_scaled, y_train_resampled)

# Calibrate Model Using Isotonic Regression
print("Calibrating model with Isotonic Regression...")
calibrated_model = CalibratedClassifierCV(final_model, method='isotonic', cv=3)
calibrated_model.fit(X_train_scaled, y_train_resampled)

# Make Predictions
print("Making predictions...")
y_pred = calibrated_model.predict(X_test_scaled)

# Evaluate Model
print("Evaluating model...")
kappa_score = cohen_kappa_score(y_test, y_pred)
print(f"Cohen's Kappa Score: {kappa_score:.4f}")

print("Generating additional evaluation metrics...")
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save Model
print("Saving model...")
joblib.dump(calibrated_model, 'stacking_model.joblib', compress=3)
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(le, 'label_encoder.joblib')
joblib.dump(top_features, 'top_features.joblib')

print("Model saved successfully!")
