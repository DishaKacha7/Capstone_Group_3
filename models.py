from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, n_jobs=-1)
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    def predict(self, X_test):
        return self.model.predict(X_test)

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    def predict(self, X_test):
        return self.model.predict(X_test)

class LightGBMModel:
    def __init__(self):
        self.model = lgb.LGBMClassifier(n_jobs=-1)
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    def predict(self, X_test):
        return self.model.predict(X_test)