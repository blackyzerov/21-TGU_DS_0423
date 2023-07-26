"""config"""
from dataclasses import dataclass, field

#from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore

from .models import ModelConfig
from .feature import FeatureConfig


@dataclass()
class DatasetConfig:
    """DatasetConfig"""
    input_data_path: str


@dataclass()
class SplittingConfig:
    """SplittingConfig"""
    test_size: float = field(default=0.25)
    random_state: int = field(default=42)


@dataclass
class TrainingPipelineConfig:
    """TrainingPipelineConfig"""
    model: ModelConfig
    dataset: DatasetConfig
    feature: FeatureConfig
    split: SplittingConfig


cs = ConfigStore.instance()
cs.store(name="base_config", node=TrainingPipelineConfig)
