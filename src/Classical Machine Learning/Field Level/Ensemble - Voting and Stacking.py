# === field_train_and_save.py ===
#!/usr/bin/env python3
"""
Standalone field‑level training + model export (NaN‑safe).

• Reads '/home/ubuntu/sai/final_data.parquet'
• Aggregates pixel rows → one record per field ('fid')
• Splits by 'fid' (80/20)
• Imputes missing values (median) and scales numeric features
• Builds One‑vs‑Rest soft‑Voting and Stacking ensembles
• Prints Accuracy & Cohen's κ, plots confusion matrices
• Saves: ensemble_voting.pkl, ensemble_stacking.pkl, label_encoder.pkl
• Also saves train/test splits to 'train_data.parquet' and 'test_data.parquet'
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.impute          import SimpleImputer
from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.multiclass      import OneVsRestClassifier
from sklearn.metrics         import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

# ─────────── CONFIG ────────────
PARQUET_PATH = "/home/ubuntu/sai/final_data.parquet"
EXCLUDE_COLS = ['id','point','fid','crop_id','SHAPE_AREA','SHAPE_LEN']
TEST_SIZE    = 0.20
SEED         = 42
# ────────────────────────────────

# 1) Load & aggregate
raw = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
print("Raw rows:", raw.shape)
feature_cols = [c for c in raw.columns if c not in EXCLUDE_COLS + ['crop_name']]

# Build mapping dict to avoid keyword-agg issues:
mapping = {c: "mean" for c in feature_cols}
mapping["crop_name"] = lambda x: x.mode().iat[0] if not x.mode().empty else None

field_df = raw.groupby("fid").agg(mapping).reset_index()
print("Aggregated (fields):", field_df.shape)

# 2) Split by field
train_fids, test_fids = train_test_split(
    field_df.fid.unique(),
    test_size=TEST_SIZE,
    random_state=SEED,
)
train_df = field_df[field_df.fid.isin(train_fids)].reset_index(drop=True)
test_df  = field_df[field_df.fid.isin(test_fids )].reset_index(drop=True)
print(f"Train fields: {train_df.shape[0]} | Test fields: {test_df.shape[0]}")
X_train_raw, X_test_raw = train_df[feature_cols], test_df[feature_cols]

# Save splits for inference
train_df.to_parquet("train_data.parquet", index=False)
test_df.to_parquet("test_data.parquet", index=False)
print("Saved train_data.parquet and test_data.parquet")

# 3) Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df.crop_name)
y_test  = le.transform(test_df.crop_name)

# 4) Preprocessing pipeline (impute + scale)
num_pipe   = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler())
])
preproc = ColumnTransformer([("num", num_pipe, feature_cols)])

# 5) Define One‑vs‑Rest base learners
base_lr  = OneVsRestClassifier(LogisticRegression(max_iter=1000))
base_rf  = OneVsRestClassifier(RandomForestClassifier(
    n_estimators=400, n_jobs=-1, random_state=SEED
))
base_hgb = OneVsRestClassifier(HistGradientBoostingClassifier(
    random_state=SEED
))
estimators = [('lr', base_lr), ('rf', base_rf), ('hgb', base_hgb)]

# optional XGBoost
try:
    from xgboost import XGBClassifier
    base_xgb = OneVsRestClassifier(
        XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=SEED
        )
    )
    estimators.append(('xgb', base_xgb))
    print("xgboost: added.")
except ImportError:
    print("xgboost: not installed, skipping.")

# 6) Build ensemble pipelines
voting_pipe = Pipeline([
    ("prep", preproc),
    ("vote", VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1))
])
stack_pipe = Pipeline([
    ("prep", preproc),
    ("stack", StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        passthrough=True,
        n_jobs=-1
    ))
])

# 7) Train, evaluate, plot
def eval_plot(name, pipe):
    pipe.fit(X_train_raw, y_train)
    preds = pipe.predict(X_test_raw)
    acc   = accuracy_score(y_test, preds)
    kappa = cohen_kappa_score(y_test, preds)
    print(f"\n{name} Accuracy: {acc:.4f} | Cohen κ: {kappa:.4f}")

    cm    = confusion_matrix(y_test, preds)
    pct   = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True) * 100)
    labels = np.array([f"{v:.1f}%" for v in pct.flatten()]).reshape(pct.shape)

    plt.figure(figsize=(9,7))
    sns.heatmap(pct, annot=labels, fmt='', cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"{name} Confusion Matrix (%)")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.xticks(rotation=45,ha='right'); plt.tight_layout(); plt.show()
    return preds

preds_v = eval_plot("Voting",   voting_pipe)
preds_s = eval_plot("Stacking", stack_pipe)

# 8) Save pipelines & encoder
joblib.dump(voting_pipe, "ensemble_voting.pkl")
joblib.dump(stack_pipe,  "ensemble_stacking.pkl")
joblib.dump(le,          "label_encoder.pkl")
print("Saved: ensemble_voting.pkl, ensemble_stacking.pkl, label_encoder.pkl")


# === field_inference_only.py ===
#!/usr/bin/env python3
"""
Field‑level inference ONLY (no training).

• Reads an unlabeled test parquet of pixel rows
• Aggregates numeric features per field by mean
• Loads saved Voting & Stacking pipelines + LabelEncoder
• Predicts and decodes crop names
• Saves:
    results_ensemble_field_voting.csv
    results_ensemble_field_stacking.csv
"""

import numpy as np
import pandas as pd
from joblib import load

# Config
TEST_PARQUET       = "/home/ubuntu/sai/final_data.parquet"
VOTE_PIPE_FILE     = "ensemble_voting.pkl"
STACK_PIPE_FILE    = "ensemble_stacking.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
OUT_VOTING         = "results_ensemble_field_voting.csv"
OUT_STACK          = "results_ensemble_field_stacking.csv"


def main():
    # Load and aggregate
    df = pd.read_parquet(TEST_PARQUET, engine="pyarrow")
    df_num = df.select_dtypes(include=[np.number])
    if 'fid' not in df_num.columns:
        raise ValueError("'fid' must be numeric")
    df_feat = df_num.groupby('fid', as_index=False).mean()

    # Prepare X and fid list
    fids = df_feat['fid'].to_numpy()
    X    = df_feat.drop(columns=['fid'])

    # Load pipelines & encoder
    vote_pipe  = load(VOTE_PIPE_FILE)
    stack_pipe = load(STACK_PIPE_FILE)
    le          = load(LABEL_ENCODER_FILE)

    # Predict and decode
    codes_v   = vote_pipe.predict(X)
    codes_s   = stack_pipe.predict(X)
    labels_v  = le.inverse_transform(codes_v)
    labels_s  = le.inverse_transform(codes_s)

    # Save
    pd.DataFrame({"fid":fids, "predicted":labels_v}).to_csv(OUT_VOTING, index=False)
    print(f"Saved → {OUT_VOTING}")
    pd.DataFrame({"fid":fids, "predicted":labels_s}).to_csv(OUT_STACK, index=False)
    print(f"Saved → {OUT_STACK}")

if __name__=="__main__":
    main()
