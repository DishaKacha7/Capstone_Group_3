#!/usr/bin/env python3
"""
Field‑level test inference: load saved test split and both ensembles,
produce two CSVs with field_id, true label, and predicted label.

• Reads 'test_data.parquet' (your saved test split)
• Loads both Voting & Stacking pipelines plus the LabelEncoder
• Predicts for each field and decodes to crop names
• Writes:
    - results_field_level_stacking_test.csv
    - results_field_level_voting_test.csv
"""

import pandas as pd
from joblib import load

# ─────────── CONFIG ────────────
TEST_PARQUET        = "test_data.parquet"
VOTING_PIPELINE     = "ensemble_voting.pkl"
STACK_PIPELINE      = "ensemble_stacking.pkl"
LABEL_ENCODER_FILE  = "label_encoder.pkl"

OUTPUT_STACK = "results_field_level_stacking_test.csv"
OUTPUT_VOTE  = "results_field_level_voting_test.csv"
# ────────────────────────────────

def main():
    # 1) Load the saved field‑level test split
    df_test = pd.read_parquet(TEST_PARQUET)
    print("Loaded test split:", df_test.shape)

    # 2) Extract features, IDs, and ground truth
    fids        = df_test['fid'].to_numpy()
    true_labels = df_test['crop_name'].to_numpy()
    X_test      = df_test.drop(columns=['fid', 'crop_name'])

    # 3) Load both pipelines and the label encoder
    voting_pipe = load(VOTING_PIPELINE)
    stacking_pipe = load(STACK_PIPELINE)
    le = load(LABEL_ENCODER_FILE)
    print("Loaded pipelines and LabelEncoder.")

    # 4) Make predictions
    codes_stack = stacking_pipe.predict(X_test)
    codes_vote  = voting_pipe.predict(X_test)

    # 5) Decode numeric codes to crop names
    preds_stack = le.inverse_transform(codes_stack)
    preds_vote  = le.inverse_transform(codes_vote)

    # 6) Build and save the stacking results CSV
    df_stack = pd.DataFrame({
        'fid':             fids,
        'true_label':      true_labels,
        'predicted_label': preds_stack
    })
    df_stack.to_csv(OUTPUT_STACK, index=False)
    print(f"Saved stacking results → {OUTPUT_STACK}")

    # 7) Build and save the voting results CSV
    df_vote = pd.DataFrame({
        'fid':             fids,
        'true_label':      true_labels,
        'predicted_label': preds_vote
    })
    df_vote.to_csv(OUTPUT_VOTE, index=False)
    print(f"Saved voting results  → {OUTPUT_VOTE}")

if __name__ == '__main__':
    main()
