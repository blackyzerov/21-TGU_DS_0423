"""build_features"""
from typing import List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    """process_categorical_features"""
    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(categorical_df).toarray())


def build_categorical_pipeline() -> Pipeline:
    """build_categorical_pipeline"""
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    """process_numerical_features"""
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def build_numerical_pipeline() -> Pipeline:
    """build_numerical_pipeline"""
    num_pipeline = Pipeline(
        [
            ("outliers", OutlierRemover()),
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
        ]
    )
    return num_pipeline


def make_features(transformer: ColumnTransformer, df_1: pd.DataFrame) -> pd.DataFrame:
    """make_features"""
    return pd.DataFrame(transformer.transform(df_1))


def extract_target(df_1: pd.DataFrame, target_col: List[str]) -> pd.Series:
    """extract_target"""
    target = df_1[target_col].values
    return target


class OutlierRemover(BaseEstimator, TransformerMixin):
    """OutlierRemover"""
    def __init__(self, factor=1.5):
        self.factor = factor

    def outlier_removal(self, x_1: pd.DataFrame):
        """outlier_removal"""
        x_1 = pd.Series(x_1).copy()
        q_1 = x_1.quantile(0.25)
        q_3 = x_1.quantile(0.75)
        iqr = q_3 - q_1
        lower_bound = q_1 - (self.factor * iqr)
        upper_bound = q_3 + (self.factor * iqr)
        x_1.loc[((x_1 < lower_bound) | (x_1 > upper_bound))] = np.nan
        return pd.Series(x_1)

    def fit(self, x_1, y_1=None):
        """fit"""
        return self

    def transform(self, x_1: np.array):
        """transform"""
        return pd.DataFrame(x_1).apply(self.outlier_removal)


def build_transformer(categorical_features: List[str],
                      numerical_features: List[str]) -> ColumnTransformer:
    """build_transformer"""
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                list(categorical_features),
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                list(numerical_features),
            ),
        ]
    )
    return transformer
