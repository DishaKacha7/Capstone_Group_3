import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

class FeatureSelector:
    def __init__(self, threshold=0.01, top_n=30):
        self.threshold = threshold
        self.top_n = top_n
        self.selected_features = None

    def variance_thresholding(self, X):
        selector = VarianceThreshold(threshold=self.threshold)
        return X.loc[:, selector.fit(X).get_support()]

    def feature_importance(self, X, y):
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        self.selected_features = feature_importance['Feature'][:self.top_n].tolist()
        return X[self.selected_features]

    def mutual_info_selection(self, X, y):
        mi = mutual_info_classif(X, y, random_state=42)
        mi_df = pd.DataFrame({'Feature': X.columns, 'Score': mi})
        mi_df = mi_df.sort_values(by='Score', ascending=False)
        self.selected_features = mi_df['Feature'][:self.top_n].tolist()
        return X[self.selected_features]

    def select_features(self, X, y):
        X = self.variance_thresholding(X)
        X = self.feature_importance(X, y)
        return X
