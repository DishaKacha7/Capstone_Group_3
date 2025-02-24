from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, classification_report
import joblib
import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class ModelEvaluator:
    def __init__(self):
        self.scaler = StandardScaler()

    def train_and_evaluate(self, models, X_train, X_test, y_train, y_test, le):
        print("Scaling data...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        results = {}
        for name, model in models.items():
            print(f"Training and evaluating {name}...")
            model.train(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            conf_matrix = confusion_matrix(y_test, y_pred)
            row_sums = conf_matrix.sum(axis=1, keepdims=True)
            conf_matrix_pct = (conf_matrix / row_sums) * 100
            conf_matrix_pct = np.nan_to_num(conf_matrix_pct)  # Handle division by zero
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            print(f"Cohen's Kappa Score: {cohen_kappa_score(y_test, y_pred):.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=le.classes_))
            plt.figure(figsize=(10, 8))
            labels = [f'{val:.1f}%' for val in conf_matrix_pct.flatten()]
            labels = np.array(labels).reshape(conf_matrix_pct.shape)
            sns.heatmap(conf_matrix_pct, annot=labels, fmt='', cmap='Blues', xticklabels=le.classes_,
                        yticklabels=le.classes_)
            plt.title(f'Confusion Matrix  - {name}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'cohen_kappa': cohen_kappa_score(y_test, y_pred),
                'confusion_matrix': conf_matrix,
                'confusion_matrix_pct': conf_matrix_pct,
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