
"""
preprocess.py
Handles feature engineering, cleaning, and the full preprocessing pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import os
import sys

# Local import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_data

# Constants
TARGET = 'TARGET'
ID_COL = 'SK_ID_CURR'
MISSING_THRESHOLD = 40
RARE_CATEGORY_THRESHOLD = 0.005

def clean_and_create_features(df: pd.DataFrame):
    """
    Clean dataset and create useful engineered features.
    """
    
    df = df.copy()

    # ---------------------------------------------------------
    # Drop ID column
    # ---------------------------------------------------------
    if ID_COL in df.columns:
        df.drop(ID_COL, axis=1, inplace=True)

    # ---------------------------------------------------------
    # Drop columns with too many missing values
    # ---------------------------------------------------------
    missing_pct = df.isnull().mean() * 100
    drop_cols = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()

    print(f"🔴 Dropping {len(drop_cols)} columns with >{MISSING_THRESHOLD}% missing")

    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # ---------------------------------------------------------
    # DAYS_EMPLOYED anomaly handling
    # ---------------------------------------------------------
    if 'DAYS_EMPLOYED' in df.columns:

        # anomaly flag
        df['DAYS_EMPLOYED_ANOM'] = (
            df['DAYS_EMPLOYED'] == 365243
        ).astype(int)

        # replace anomaly with NaN
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(
            365243,
            np.nan
        )

    # ---------------------------------------------------------
    # Feature Engineering
    # ---------------------------------------------------------

    if all(col in df.columns for col in ['AMT_CREDIT', 'AMT_INCOME_TOTAL']):
        df['CREDIT_INCOME_RATIO'] = (
            df['AMT_CREDIT'] /
            (df['AMT_INCOME_TOTAL'] + 1e-5)
        )

    if all(col in df.columns for col in ['AMT_ANNUITY', 'AMT_INCOME_TOTAL']):
        df['ANNUITY_INCOME_RATIO'] = (
            df['AMT_ANNUITY'] /
            (df['AMT_INCOME_TOTAL'] + 1e-5)
        )

    if all(col in df.columns for col in ['AMT_ANNUITY', 'AMT_CREDIT']):
        df['CREDIT_TERM'] = (
            df['AMT_ANNUITY'] /
            (df['AMT_CREDIT'] + 1e-5)
        )

    # ---------------------------------------------------------
    # Separate target
    # ---------------------------------------------------------
    if TARGET in df.columns:
        y = df[TARGET].copy()
        X = df.drop(TARGET, axis=1)
        return X, y

    return df, None


class RareCategoryAggregator(BaseEstimator, TransformerMixin):
    """
    Groups rare categories into 'Other'
    """

    def __init__(self, threshold=RARE_CATEGORY_THRESHOLD):
        self.threshold = threshold
        self.category_mappings_ = {}

    def fit(self, X, y=None):

        X = pd.DataFrame(X)

        for col in X.columns:
            freq = X[col].value_counts(normalize=True)

            keep = freq[freq >= self.threshold].index.tolist()

            self.category_mappings_[col] = keep

        return self

    def transform(self, X):

        X = pd.DataFrame(X).copy()

        for col in X.columns:

            if col in self.category_mappings_:

                keep = self.category_mappings_[col]

                X[col] = X[col].apply(
                    lambda x: x if x in keep else 'Other'
                )

        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features, dtype=object)


def build_preprocessing_pipeline():
    """
    Build preprocessing pipeline
    """

    from sklearn.compose import make_column_selector

    # ---------------------------------------------------------
    # Numeric pipeline
    # ---------------------------------------------------------
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # ---------------------------------------------------------
    # Categorical pipeline
    # ---------------------------------------------------------
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('rare_agg', RareCategoryAggregator()),
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=True
        ))
    ])

    # ---------------------------------------------------------
    # Column transformer
    # ---------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            (
                'num',
                numeric_transformer,
                make_column_selector(dtype_include=np.number)
            ),
            (
                'cat',
                categorical_transformer,
                make_column_selector(dtype_include='object')
            )
        ],
        remainder='drop',
        n_jobs=-1
    )

    return preprocessor


def build_training_pipeline(model, preprocessor):
    """
    preprocessing -> SMOTE -> classifier
    """

    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])

    return pipeline


def prepare_data(random_state=42, test_size=0.2):

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    data_path = os.path.join(
        BASE_DIR,
        "data",
        "raw",
        "application_train.csv"
    )

    df = load_data(data_path)

    # clean + engineer
    X, y = clean_and_create_features(df)

    print(f"📊 Features shape after cleaning: {X.shape}")

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"📦 Train size: {X_train.shape}")
    print(f"📦 Test size: {X_test.shape}")

    print(f"📦 Train default rate: {y_train.mean():.2%}")
    print(f"📦 Test default rate: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = prepare_data()

    preprocessor = build_preprocessing_pipeline()

    X_train_processed = preprocessor.fit_transform(X_train)

    print(f"✅ Processed train shape: {X_train_processed.shape}")

    X_test_processed = preprocessor.transform(X_test)

    print(f"✅ Processed test shape: {X_test_processed.shape}")

