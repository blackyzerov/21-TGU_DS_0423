"""predict"""
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

SklearnRegressionModel = Union[RandomForestRegressor, LogisticRegression]


def predict_model(
    model: SklearnRegressionModel, features: pd.DataFrame
) -> np.ndarray:
    """Predict_model"""
    predicts = model.predict(features)
    return predicts


def evaluate_model(
    predicts: np.ndarray, target: pd.Series
) -> Dict[str, float]:
    """evaulate_model"""
    return {
        "accuracy": accuracy_score(target, predicts),
    }
