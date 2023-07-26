"""test_make_dataset"""
import pandas as pd

from heart.data.make_dataset import read_data, split_train_test_data
# from heart.entities import SplittingConfig


def test_load_dataset(dataset_path: str, target_col: str):
    data = read_data(dataset_path)
    assert len(data) == 100
    assert target_col in data.keys(), (
            "target_col not in dataset"
        )


def test_split_dataset(dataset: pd.DataFrame):
    """Function test_split_dataset"""
    train_df, test_df = split_train_test_data(dataset, test_size=0.25, random_state=42)
    assert train_df.shape[0] == 75
    assert test_df.shape[0] == 25