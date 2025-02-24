import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

class HyperparameterTuner:
    def __init__(self, models, n_trials=1):
        self.models = models
        self.n_trials = n_trials
        self.best_models = {}

    def tune_rf(self, X, y):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                'max_depth': trial.suggest_int('max_depth', 5, 30, step=5),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20, step=2),
            }
            model = RandomForestClassifier(**params, n_jobs=-1, random_state=42)
            return cross_val_score(model, X, y, cv=3, scoring='accuracy', n_jobs=-1).mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        best_params = study.best_params
        return RandomForestClassifier(**best_params, n_jobs=-1, random_state=42)

    def tune_lgb(self, X, y):
        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 10, 100, step=10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            }
            model = lgb.LGBMClassifier(**params, n_jobs=-1)
            return cross_val_score(model, X, y, cv=3, scoring='accuracy', n_jobs=-1).mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        best_params = study.best_params
        return lgb.LGBMClassifier(**best_params, n_jobs=-1)

    def tune_xgb(self, X, y):
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 18, step=3),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            }
            model = xgb.XGBClassifier(**params, n_jobs=-1, use_label_encoder=False, eval_metric='logloss', verbosity=0)
            return cross_val_score(model, X, y, cv=3, scoring='accuracy', n_jobs=-1).mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        best_params = study.best_params
        return xgb.XGBClassifier(**best_params, n_jobs=-1, use_label_encoder=False, eval_metric='logloss', verbosity=0)

    def tune_hyperparameters(self, X, y):
        if 'Random Forest' in self.models:
            self.best_models['Random Forest'] = self.tune_rf(X, y)
        if 'LightGBM' in self.models:
            self.best_models['LightGBM'] = self.tune_lgb(X, y)
        if 'XGBoost' in self.models:
            self.best_models['XGBoost'] = self.tune_xgb(X, y)
        return self.best_models
