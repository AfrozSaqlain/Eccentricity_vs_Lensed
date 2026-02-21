import argparse
import json
import math
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

from gwtorch.modules.data_loading import (
    NpyGWDataset,
    collate_with_weights,
    default_transform_224,
    safe_num_workers,
)
from gwtorch.modules.ml_utils import seed_everything
from gwtorch.modules.neural_net import (
    CNNBinaryModel,
    CNN_Model,
    SimpleCNN,
    SimpleCNNBinary,
    create_transformer_model,
)
from gwtorch.modules.pathing import class_from_file_path, default_meta_csv_path, discover_npy_files


TASK_TO_CLASSES = {
    "cnn": {"lensed": 0, "eccentric": 1, "unlensed": 2},
    "vit": {"lensed": 0, "eccentric": 1, "unlensed": 2},
    "binary_cnn": {"lensed": 0, "eccentric": 1},
    "binary_vit": {"lensed": 0, "eccentric": 1},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Random-search hyperparameter tuning for GWtorch models.")
    parser.add_argument("--task", type=str, required=True, choices=["cnn", "binary_cnn", "vit", "binary_vit"])
    parser.add_argument("--model", type=str, required=True, help="Specific architecture to tune")
    parser.add_argument("--path-name", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size-choices", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-max", type=float, default=5e-4)
    parser.add_argument("--gamma-min", type=float, default=0.5)
    parser.add_argument("--gamma-max", type=float, default=0.95)
    parser.add_argument("--dropout-min", type=float, default=0.1)
    parser.add_argument("--dropout-max", type=float, default=0.5)
    parser.add_argument("--weight-decay-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay-max", type=float, default=1e-3)
    parser.add_argument("--save-best-model", action="store_true")
    parser.add_argument("--results-dir", type=str, default="results/hyperparameter_tuning")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name under <task>_<model>/")
    parser.add_argument("--overwrite-run", action="store_true", help="Allow writing into an existing run directory.")
    return parser.parse_args()


class GWDataset(Dataset):
    def __init__(self, file_list, label_map, transform=None, meta_csv="data"):
        meta_csv_str = str(meta_csv)
        if Path(meta_csv_str).suffix != ".csv":
            meta_csv_str = str(Path("results") / f"{meta_csv}_parameters.csv")
        self.dataset = NpyGWDataset(
            file_list=file_list,
            label_map=label_map,
            transform=transform,
            meta_csv=meta_csv_str,
            return_lnB=False,
            weight=1.0,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def custom_collate_fn(batch):
    return collate_with_weights(batch)


def validate_model_for_task(task, model):
    allowed = {
        "cnn": {"CNN", "SimpleCNN"},
        "binary_cnn": {"CNN", "SimpleCNN"},
        "vit": {"ViT", "ViTLinear", "CvT"},
        "binary_vit": {"ViT", "ViTLinear", "CvT"},
    }
    if model not in allowed[task]:
        raise ValueError(f"Model '{model}' is not valid for task '{task}'. Allowed: {sorted(allowed[task])}")


def filter_files(path_name, task):
    all_files = discover_npy_files(path_name)
    if not all_files:
        raise FileNotFoundError(f"No .npy files found under {path_name}/*/*.npy")
    if "binary" in task:
        all_files = [p for p in all_files if class_from_file_path(p) in {"eccentric", "lensed"}]
        if not all_files:
            raise FileNotFoundError(f"No eccentric/lensed .npy files found under {path_name}")
    return all_files


def compute_class_weights(file_list, label_map):
    labels = [label_map[class_from_file_path(p)] for p in file_list]
    counts = Counter(labels)
    total = sum(counts.values())
    return torch.tensor([total / counts[i] for i in range(len(label_map))], dtype=torch.float32)


def build_loaders(path_name, seed, batch_size, task):
    label_map = TASK_TO_CLASSES[task]
    all_files = filter_files(path_name, task)
    all_labels = [class_from_file_path(path) for path in all_files]

    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels, test_size=0.3, stratify=all_labels, random_state=seed
    )
    val_files, _test_files, *_ = train_test_split(
        temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=seed
    )

    class_weights = compute_class_weights(train_files, label_map)
    transform = default_transform_224(antialias=False)
    meta_csv = default_meta_csv_path(path_name)

    train_data = GWDataset(train_files, label_map=label_map, transform=transform, meta_csv=meta_csv)
    val_data = GWDataset(val_files, label_map=label_map, transform=transform, meta_csv=meta_csv)

    num_channels = train_data[0][0].shape[0]
    num_workers = safe_num_workers()
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
    )
    return num_channels, train_loader, val_loader, class_weights


def create_model(task, model_name, num_channels, dropout, device):
    if task == "cnn":
        if model_name == "CNN":
            return CNN_Model(input_channels=num_channels, dropout=dropout).to(device)
        return SimpleCNN(input_channels=num_channels, dropout=dropout).to(device)
    if task == "binary_cnn":
        if model_name == "CNN":
            return CNNBinaryModel(input_channels=num_channels, dropout=dropout).to(device)
        return SimpleCNNBinary(input_channels=num_channels, dropout=dropout).to(device)
    if task == "vit":
        return create_transformer_model(
            name=model_name,
            num_channels=num_channels,
            device=device,
            num_classes=3,
            dropout=dropout,
        )
    if task == "binary_vit":
        return create_transformer_model(
            name=model_name,
            num_channels=num_channels,
            device=device,
            num_classes=2,
            dropout=dropout,
        )
    raise ValueError(f"Unknown task '{task}'")


def sample_hparams(args, rng):
    lr = 10 ** rng.uniform(math.log10(args.lr_min), math.log10(args.lr_max))
    gamma = rng.uniform(args.gamma_min, args.gamma_max)
    dropout = rng.uniform(args.dropout_min, args.dropout_max)
    weight_decay = 10 ** rng.uniform(math.log10(args.weight_decay_min), math.log10(args.weight_decay_max))
    batch_size = rng.choice(args.batch_size_choices)
    return {
        "lr": lr,
        "gamma": gamma,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "batch_size": int(batch_size),
    }


def run_trial(model, train_loader, val_loader, class_weights, device, epochs, lr, gamma, weight_decay):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=5, gamma=gamma)
    criterion = nn.CrossEntropyLoss(
        reduction="none",
        weight=class_weights.to(device),
        label_smoothing=0.1,
    )

    best_val_acc = -1.0
    final_val_loss = None
    for _ in range(epochs):
        model.train()
        for data, labels, weights, *_ in train_loader:
            data, labels, weights = data.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad()
            output = model(data)
            unweighted_loss = criterion(output, labels)
            loss = (unweighted_loss * weights).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for data, labels, *_ in val_loader:
                data, labels = data.to(device), labels.to(device)
                output = model(data)
                loss = criterion(output, labels).mean()
                acc = (output.argmax(1) == labels).float().mean()
                val_loss += loss.item()
                val_acc += acc.item()
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        final_val_loss = val_loss
        best_val_acc = max(best_val_acc, val_acc)
        scheduler.step()

    return best_val_acc, final_val_loss


def main():
    args = parse_args()
    validate_model_for_task(args.task, args.model)
    seed_everything(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = Path(args.results_dir) / f"{args.task}_{args.model}"
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / run_name
    if output_dir.exists() and not args.overwrite_run:
        raise FileExistsError(
            f"Run directory already exists: {output_dir}. Use --overwrite-run or choose --run-name."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_file = output_dir / "trials.jsonl"
    best_model_path = output_dir / "best_model.pth"

    print(f"Using device: {device}")
    print(f"Tuning task={args.task}, model={args.model}, trials={args.trials}, run={run_name}")

    best = {"trial": -1, "val_acc": -1.0}
    for trial in range(1, args.trials + 1):
        hp = sample_hparams(args, rng)
        num_channels, train_loader, val_loader, class_weights = build_loaders(
            path_name=args.path_name,
            seed=args.seed,
            batch_size=hp["batch_size"],
            task=args.task,
        )
        model = create_model(
            task=args.task,
            model_name=args.model,
            num_channels=num_channels,
            dropout=hp["dropout"],
            device=device,
        )
        val_acc, val_loss = run_trial(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            device=device,
            epochs=args.epochs,
            lr=hp["lr"],
            gamma=hp["gamma"],
            weight_decay=hp["weight_decay"],
        )

        record = {
            "trial": trial,
            "task": args.task,
            "model": args.model,
            "epochs": args.epochs,
            "val_acc": float(val_acc),
            "val_loss": float(val_loss),
            **hp,
        }
        with open(trials_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[Trial {trial:03d}] val_acc={val_acc:.4f} val_loss={val_loss:.4f} hp={hp}")

        if val_acc > best["val_acc"]:
            best = record
            if args.save_best_model:
                torch.save(model.state_dict(), best_model_path)

    with open(output_dir / "best_result.json", "w") as f:
        json.dump(best, f, indent=2)

    print("\nBest trial summary:")
    print(json.dumps(best, indent=2))
    print(f"All trials saved to: {trials_file}")
    if args.save_best_model:
        print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
