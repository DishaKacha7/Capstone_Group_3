import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self, exclude_cols=None):
        if exclude_cols is None:
            exclude_cols = ['id', 'point', 'fid', 'crop_id', 'SHAPE_AREA', 'SHAPE_LEN']
        self.exclude_cols = exclude_cols
        self.label_encoder = LabelEncoder()

    def prepare_data(self, df):
        feature_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in feature_cols if col not in self.exclude_cols and not df[col].isna().any()]
        X = df[feature_cols]
        y_encoded = self.label_encoder.fit_transform(df['crop_name'])
        return X, y_encoded, self.label_encoder, feature_cols
