import pandas as pd
from data_processing import DataPreprocessor
from data_splitting import DataSplitter
from models import XGBoostModel
from model_utils import ModelEvaluator, ModelSaver

def main():
    file_path = "/home/ubuntu/dev/Data/final_data.parquet"
    data = pd.read_parquet(file_path, engine="pyarrow").sample(frac=1, random_state=42)

    # Split into train and test sets
    train_df, test_df = DataSplitter.train_test_split_by_field(data)

    # Define the top 15 features
    top_15_features = [
        "EVI_B11_standard_deviation", "B12_B12_variance", "B11_B11_variance", "B2_B2_variance",
        "EVI_EVI_standard_deviation", "EVI_B6_standard_deviation", "EVI_B2_standard_deviation",
        "B12_B12_quantile_q_0.05", "B11_B11_kurtosis", "B11_B11_minimum", "B6_B6_variance",
        "B2_B2_quantile_q_0.05", "B6_B6_mean_change", "EVI_B12_standard_deviation", "B6_B6_quantile_q_0.95"
    ]

    # Preprocess the data
    preprocessor = DataPreprocessor()
    X_train, y_train, le, _ = preprocessor.prepare_data(train_df)
    X_test, y_test, _, _ = preprocessor.prepare_data(test_df)

    # Select only the top 15 features
    X_train = X_train[top_15_features]
    X_test = X_test[top_15_features]

    # Initialize XGBoost Model
    xgb_model = XGBoostModel()

    # Run Hyperparameter Tuning
    xgb_model.tune_hyperparameters(X_train, y_train, n_trials=30)

    # Train and Evaluate
    models = {"XGBoost": xgb_model}
    evaluator = ModelEvaluator()
    results, scaler = evaluator.train_and_evaluate(models, X_train, X_test, y_train, y_test, le)

    # Save Results
    save_path = ModelSaver.save_models(results, scaler, le, top_15_features)
    print(f"Models saved in: {save_path}")

if __name__ == "__main__":
    main()


