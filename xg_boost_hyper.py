import xgboost as xgb
import optuna
from sklearn.model_selection import cross_val_score


class XGBoostModel:
    def __init__(self):
        self.model = None
        self.best_params = None

    def objective(self, trial, X_train, y_train):
        print("Evaluating trial with new set of hyperparameters...")
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10),
        }
        print(f"Trying parameters: {params}")

        model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss", n_jobs=-1)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        mean_score = scores.mean()
        print(f"Trial completed with accuracy: {mean_score:.4f}")
        return mean_score

    def tune_hyperparameters(self, X_train, y_train, n_trials=30):
        print("\nStarting XGBoost Hyperparameter Tuning...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=n_trials)

        self.best_params = study.best_params
        print("\nBest XGBoost Hyperparameters Found:")
        print(self.best_params)

        self.model = xgb.XGBClassifier(**self.best_params, use_label_encoder=False, eval_metric="logloss", n_jobs=-1)

    def train(self, X_train, y_train):
        print("\nTraining XGBoost with Best Parameters...")
        self.model.fit(X_train, y_train)
        print("XGBoost Training Completed.")

    def predict(self, X_test):
        print("Making predictions with XGBoost...")
        return self.model.predict(X_test)
