from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from gwtorch.modules.pathing import class_from_file_path, discover_npy_files, sample_id_from_file_path


def safe_num_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def default_transform_224(antialias: bool = True):
    return transforms.Compose([
        transforms.Resize((224, 224), antialias=antialias),
    ])


def resolve_meta_csv_path(meta_csv: Optional[str], path_name: Optional[str] = None) -> Optional[Path]:
    if meta_csv:
        return Path(meta_csv)
    if path_name:
        dataset_name = Path(path_name).resolve().name
        return Path("results") / f"{dataset_name}_parameters.csv"
    return None


def load_metadata(meta_csv: Optional[str], path_name: Optional[str] = None) -> Optional[pd.DataFrame]:
    meta_csv_path = resolve_meta_csv_path(meta_csv=meta_csv, path_name=path_name)
    if meta_csv_path is None:
        return None
    try:
        df = pd.read_csv(meta_csv_path)
        if "sample" in df.columns:
            df = df.set_index("sample")
        return df
    except FileNotFoundError:
        return None


def build_file_list(path_name: str, include_classes: Optional[set[str]] = None) -> list[str]:
    files = discover_npy_files(path_name)
    if include_classes is None:
        return files
    return [p for p in files if class_from_file_path(p) in include_classes]


def lnB_for_sample(df: Optional[pd.DataFrame], file_path: str, label_str: str) -> float:
    if df is None:
        return 0.0

    file_number = sample_id_from_file_path(file_path)
    if file_number not in df.index:
        return 0.0

    if label_str == "lensed" and "lens_lnB" in df.columns:
        return float(df.loc[file_number, "lens_lnB"])
    if label_str == "eccentric" and "ecc_lnB" in df.columns:
        return float(df.loc[file_number, "ecc_lnB"])
    if label_str == "unlensed":
        return 100.0
    return 0.0


class NpyGWDataset(Dataset):
    def __init__(
        self,
        file_list: list[str],
        label_map: dict[str, int],
        transform=None,
        meta_csv: Optional[str] = None,
        path_name: Optional[str] = None,
        return_lnB: bool = False,
        weight: float = 1.0,
    ):
        self.file_list = file_list
        self.label_map = label_map
        self.transform = transform
        self.return_lnB = return_lnB
        self.weight = weight
        self.df = load_metadata(meta_csv=meta_csv, path_name=path_name)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        arr = np.load(file_path).astype(np.float32)
        arr = torch.from_numpy(arr)

        if self.transform:
            arr = self.transform(arr)

        label_str = class_from_file_path(file_path)
        label = self.label_map[label_str]

        if self.return_lnB:
            ln_B = lnB_for_sample(self.df, file_path, label_str)
            return arr, label, ln_B, file_path

        return arr, label, self.weight, file_path


def collate_with_weights(batch):
    images, labels, weights, file_paths = zip(*batch)
    return torch.stack(images), torch.tensor(labels), torch.tensor(weights, dtype=torch.float32), file_paths


def collate_with_lnb(batch):
    images, labels, lnBs, file_paths = zip(*batch)
    return torch.stack(images), torch.tensor(labels), lnBs, file_paths

