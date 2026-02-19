from __future__ import annotations

import glob
from pathlib import Path


def class_from_file_path(file_path: str) -> str:
    return Path(file_path).parent.name.lower()


def sample_id_from_file_path(file_path: str) -> int:
    return int(Path(file_path).stem)


def dataset_name_from_path(path_name: str) -> str:
    return Path(path_name).resolve().name


def default_meta_csv_path(path_name: str) -> Path:
    dataset_name = dataset_name_from_path(path_name)
    return Path("results") / f"{dataset_name}_parameters.csv"


def discover_npy_files(path_name: str) -> list[str]:
    return sorted(glob.glob(f"{path_name}/*/*.npy"))

