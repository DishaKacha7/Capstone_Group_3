import pandas as pd
from data_processing import DataPreprocessor
from data_splitting import DataSplitter
from models import LogisticRegressionModel, RandomForestModel, LightGBMModel,XGBoostModel
from model_utils import ModelEvaluator, ModelSaver
# from xgboost_optimizer import optimize_xgboost

def main():
    file_path = "/home/ubuntu/sai/final_data.parquet"
    data = pd.read_parquet(file_path, engine="pyarrow").sample(frac=1, random_state=42)
    print(data.head())
    # print(data.isna().sum())


    # Assuming df is your DataFrame
    exclude_cols = ['id', 'point', 'fid', 'crop_id', 'SHAPE_AREA', 'SHAPE_LEN']
    feature_cols = [col for col in data.columns if col not in exclude_cols + ['crop_name']]

    # Aggregation: average for features, most frequent for crop_name
    aggregated_df = data.groupby('fid').agg(
        {**{col: 'mean' for col in feature_cols}, 'crop_name': lambda x: x.mode()[0] if not x.mode().empty else None}
    ).reset_index()

    # Show the head of the aggregated DataFrame
    print(aggregated_df.head())
    print(aggregated_df.shape)

    field_level_data=aggregated_df.copy()
    train_df, test_df = DataSplitter.train_test_split_by_field(field_level_data)
    preprocessor = DataPreprocessor()
    X_train, y_train, le, feature_cols = preprocessor.prepare_data(train_df)
    X_test, y_test, _, _ = preprocessor.prepare_data(test_df)
    models = {
        'Logistic Regression': LogisticRegressionModel(),
        'Random Forest': RandomForestModel(),
        'LightGBM': LightGBMModel(),
        'XGBoost': XGBoostModel()
    }
    evaluator = ModelEvaluator()
    results, scaler = evaluator.train_and_evaluate(models, X_train, X_test, y_train, y_test, le)
    save_path = ModelSaver.save_models(results, scaler, le, feature_cols)
    print(f"Models saved in: {save_path}")

if __name__ == "__main__":
    main()
