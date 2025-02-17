import numpy as np

class DataSplitter:
    @staticmethod
    def train_test_split_by_field(df, test_size=0.2, random_state=42):
        unique_fids = df['fid'].unique()
        np.random.seed(random_state)
        test_fids = np.random.choice(unique_fids, size=int(len(unique_fids) * test_size), replace=False)
        test_mask = df['fid'].isin(test_fids)
        return df[~test_mask], df[test_mask]
