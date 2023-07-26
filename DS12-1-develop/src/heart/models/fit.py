"""fit"""
import pickle
from typing import Union

import pandas as pd
from hydra.utils import instantiate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


SklearnRegressionModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    model_params, train_features: pd.DataFrame, target: pd.Series
) -> SklearnRegressionModel:
    """train_model"""
    model = instantiate(model_params).fit(train_features, target.ravel())
    return model


def serialize_model(model: SklearnRegressionModel, output: str) -> str:
    """serialize_model"""
    with open(output, "wb") as f_1:
        pickle.dump(model, f_1)
    return output
