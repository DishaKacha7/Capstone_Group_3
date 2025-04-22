"""
Field‑level test inference
"""

import pandas as pd
from joblib import load

# Set paths
TEST_PARQUET        = "test_data.parquet"
VOTING_PIPELINE     = "ensemble_voting.pkl"
STACK_PIPELINE      = "ensemble_stacking.pkl"
LABEL_ENCODER_FILE  = "label_encoder.pkl"

OUTPUT_STACK = "results_field_level_stacking_test.csv"
OUTPUT_VOTE  = "results_field_level_voting_test.csv"

def main():
    df_test = pd.read_parquet(TEST_PARQUET)
    print("Loaded test split:", df_test.shape)

    fids        = df_test['fid'].to_numpy()
    true_labels = df_test['crop_name'].to_numpy()
    X_test      = df_test.drop(columns=['fid', 'crop_name'])

    # Loading encoders and models
    voting_pipe = load(VOTING_PIPELINE)
    stacking_pipe = load(STACK_PIPELINE)
    le = load(LABEL_ENCODER_FILE)
    print("Loaded pipelines and LabelEncoder.")
    codes_stack = stacking_pipe.predict(X_test)
    codes_vote  = voting_pipe.predict(X_test)

    # Need to transform preds from codes to labels
    preds_stack = le.inverse_transform(codes_stack)
    preds_vote  = le.inverse_transform(codes_vote)

    # This won't work unless you have true labels in the test set
    df_stack = pd.DataFrame({
        'fid':             fids,
        'true_label':      true_labels,
        'predicted_label': preds_stack
    })
    df_stack.to_csv(OUTPUT_STACK, index=False)
    print(f"Saved stacking results → {OUTPUT_STACK}")

    df_vote = pd.DataFrame({
        'fid':             fids,
        'true_label':      true_labels,
        'predicted_label': preds_vote
    })
    df_vote.to_csv(OUTPUT_VOTE, index=False)
    print(f"Saved voting results  → {OUTPUT_VOTE}")

if __name__ == '__main__':
    main()
