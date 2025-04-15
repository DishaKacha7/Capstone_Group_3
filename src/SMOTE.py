# Import Libraries
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import joblib

# Load Data
print("Loading data...")
file_path = "final_data.parquet"
data = pd.read_parquet(file_path, engine="pyarrow").sample(frac=1, random_state=42)

# Encode Labels
print("Encoding labels...")
le = LabelEncoder()
data['crop_name_encoded'] = le.fit_transform(data['crop_name'])

# Select Features
print("Selecting features...")
exclude_cols = ['id', 'point', 'fid', 'crop_id', 'SHAPE_AREA', 'SHAPE_LEN', 'crop_name', 'crop_name_encoded']
feature_cols = [col for col in data.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
X = data[feature_cols]
y = data['crop_name_encoded']

# Handle Missing Values
print("Handling missing values...")
X = X.dropna(axis=1, how='all')
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Feature Selection
print("Removing low variance features...")
selector = VarianceThreshold(threshold=0.01)
X = X.loc[:, selector.fit(X).variances_ > 0.01]

print("Removing highly correlated features...")
corr_matrix = X.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.9)]
X.drop(columns=to_drop, inplace=True)

print("Selecting top 25 important features...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(X, y)
top_features = pd.Series(rf.feature_importances_, index=X.columns).nlargest(25).index.tolist()
X = X[top_features]

# Train-Test-Validation Split
print("Splitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Apply SMOTE
print("Applying SMOTE...")
smote = SMOTE(random_state=42, sampling_strategy='not majority')
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:", Counter(y_train_resampled))

# Scale Data
print("Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Define Base Models (Removed CatBoost)
print("Defining base models...")
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=15, class_weight="balanced", n_jobs=-1, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_jobs=-1))
]

# Train Base Models
print("Training base models...")
for name, model in base_models:
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train_resampled)

# Define Stacking Model
print("Creating stacking classifier...")
meta_model = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight="balanced", n_jobs=-1, random_state=42)
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, passthrough=True, n_jobs=-1)

print("Training stacked model...")
stacking_model.fit(X_train_scaled, y_train_resampled)

# Calibrate Model
print("Calibrating stacked model...")
calibrated_stacking_model = CalibratedClassifierCV(stacking_model, method='isotonic', cv=3)
calibrated_stacking_model.fit(X_train_scaled, y_train_resampled)

# Predictions and Evaluation
print("Evaluating model...")
y_pred = calibrated_stacking_model.predict(X_test_scaled)
kappa_score = cohen_kappa_score(y_test, y_pred)
print(f"Cohen's Kappa Score: {kappa_score:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save Models
print("Saving model...")
joblib.dump(calibrated_stacking_model, 'stacked_model_calibrated_rf.joblib', compress=3)
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(le, 'label_encoder.joblib')
joblib.dump(top_features, 'top_features.joblib')
print("Model saved successfully!")
