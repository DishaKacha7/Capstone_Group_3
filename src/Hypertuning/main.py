#%%
import pandas as pd
from data_processing import DataPreprocessor
from data_splitting import DataSplitter
from models import LogisticRegressionModel, RandomForestModel, LightGBMModel, XGBoostModel
from model_utils import ModelEvaluator, ModelSaver
from feature_selection import FeatureSelector
from hyperparameter_tuning import HyperparameterTuner

def main():
    file_path = "final_data.parquet"
    data = pd.read_parquet(file_path, engine="pyarrow").sample(frac=1, random_state=42)

    # Splitting the dataset
    train_df, test_df = DataSplitter.train_test_split_by_field(data)

    # Data Processing
    preprocessor = DataPreprocessor()
    X_train, y_train, le, feature_cols = preprocessor.prepare_data(train_df)
    X_test, y_test, _, _ = preprocessor.prepare_data(test_df)

    # Feature Selection
    selector = FeatureSelector(top_n=30)
    X_train = selector.select_features(X_train, y_train)
    X_test = X_test[selector.selected_features]

    # Define models
    models = {
        #'Random Forest': RandomForestModel(),
        #'LightGBM': LightGBMModel(),
        'XGBoost': XGBoostModel()
    }

    # Hyperparameter Tuning
    tuner = HyperparameterTuner(models)
    best_models = tuner.tune_hyperparameters(X_train, y_train)

    # Model Training and Evaluation
    evaluator = ModelEvaluator()
    results, scaler = evaluator.train_and_evaluate(best_models, X_train, X_test, y_train, y_test, le)

    # Save Models
    save_path = ModelSaver.save_models(results, scaler, le, selector.selected_features)
    print(f"Models saved in: {save_path}")

if __name__ == "__main__":
    main()
