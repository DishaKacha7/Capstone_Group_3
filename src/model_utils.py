from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import joblib
import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt


class ModelEvaluator:
    def __init__(self):
        self.scaler = StandardScaler()

    def train_and_evaluate(self, models, X_train, X_test, y_train, y_test, le):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        results = {}
        for name, model in models.items():
            model.train(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'cohen_kappa': cohen_kappa_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'predictions': y_pred
            }
        return results, self.scaler

class ModelSaver:
    @staticmethod
    def save_models(results, scaler, le, feature_cols, save_dir='crop_classification_models'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, timestamp)
        os.makedirs(save_path, exist_ok=True)
        for name, result in results.items():
            joblib.dump(result['model'], os.path.join(save_path, f'{name.lower().replace(" ", "_")}.joblib'))
        joblib.dump(scaler, os.path.join(save_path, 'scaler.joblib'))
        joblib.dump(le, os.path.join(save_path, 'label_encoder.joblib'))
        joblib.dump(feature_cols, os.path.join(save_path, 'feature_columns.joblib'))
        return save_path
