import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:
    def __init__(self, target_column):
        self.target_column = target_column
        self.pipeline = None
        self.label_encoder = None   # 🔥 NEW

    def fit_transform(self, df):
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # 🔥 Encode target if categorical
        if y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)

        # Detect feature types
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = X.select_dtypes(include=["object"]).columns

        # Pipelines
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.pipeline = ColumnTransformer([
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features)
        ])

        X_processed = self.pipeline.fit_transform(X)

        return X_processed, y