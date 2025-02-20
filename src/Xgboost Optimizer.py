from sklearn.model_selection import GridSearchCV
import xgboost as xgb


def optimize_xgboost(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

    grid_search.fit(X_train, y_train)

    print("Best parameters found:", grid_search.best_params_)
    return grid_search.best_estimator_
