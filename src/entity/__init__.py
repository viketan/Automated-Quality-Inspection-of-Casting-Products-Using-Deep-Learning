from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    data_dir: Path
    model_filename: str

@dataclass(frozen=True)
class TrainingParam:
    img_size : int
    batch_size: int
    channel: int
    epochs: int
    activation_function_hidden: str
    activation_function_output: str
    optimizer: str
    loss: str
    metrics: str

@dataclass(frozen=True)
class EvaluationConfig:
  model_path: Path
  root_dir: Path
  mlflow_uri: str
  data_dir: Path

@dataclass(frozen=True)
class PredictionConfig:
  model_path: Path
  root_dir: Path
  img_size: int
  class_names: list
    