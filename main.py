import pandas as pd
from data_processing import DataPreprocessor
from data_splitting import DataSplitter
from models import LogisticRegressionModel, RandomForestModel, LightGBMModel
from model_utils import ModelEvaluator, ModelSaver

def main():
    file_path = "/home/ubuntu/dev/Data/final_data.parquet"
    data = pd.read_parquet(file_path, engine="pyarrow").sample(frac=1, random_state=42).head(1000000)
    train_df, test_df = DataSplitter.train_test_split_by_field(data)
    preprocessor = DataPreprocessor()
    X_train, y_train, le, feature_cols = preprocessor.prepare_data(train_df)
    X_test, y_test, _, _ = preprocessor.prepare_data(test_df)
    models = {
        'Logistic Regression': LogisticRegressionModel(),
        'Random Forest': RandomForestModel(),
        'LightGBM': LightGBMModel()
    }
    evaluator = ModelEvaluator()
    results, scaler = evaluator.train_and_evaluate(models, X_train, X_test, y_train, y_test, le)
    save_path = ModelSaver.save_models(results, scaler, le, feature_cols)
    print(f"Models saved in: {save_path}")

if __name__ == "__main__":
    main()
