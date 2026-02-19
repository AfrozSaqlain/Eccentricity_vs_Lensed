import os
import glob
import random
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from gwtorch.modules.pathing import class_from_file_path, discover_npy_files

from linformer import Linformer
from vit_pytorch.efficient import ViT
from vit_pytorch.cvt import CvT

# ---- GW utility imports ----
from gwtorch.modules.general_utils import (
    compute_roc_auc_with_misclassifications,
    plot_roc_curves,
    plot_confusion_matrix,
    plot_training_curves,
)

# -------------------------------------------------------------------------
#                            ARGUMENTS & SETUP
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Hierarchical Transformer for GW classification")
    parser.add_argument('--model', type=str, default='ViT', choices=['CvT', 'ViT'], help='Model to use')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, help='LR scheduler gamma')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model-path', type=str, default=None, help='Optional pretrained model path')
    parser.add_argument('--path-name', type=str, default='data', help='Path to dataset directory')
    parser.add_argument('--no-inference', action='store_true', help='Skip inference demo after training')
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def setup_directories():
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results/stage1_results/Plots", exist_ok=True)
    os.makedirs("./results/stage2_results/Plots", exist_ok=True)


# -------------------------------------------------------------------------
#                            DATASETS
# -------------------------------------------------------------------------

class UnlensedVsOthersDataset(Dataset):
    """Stage 1: unlensed (0) vs {lensed, eccentric} (1)"""
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        arr = torch.from_numpy(np.load(file_path).astype(np.float32))
        if self.transform: arr = self.transform(arr)
        label_str = class_from_file_path(file_path)
        label = 0 if label_str == "unlensed" else 1
        return arr, label, 1.0, file_path


class LensedVsEccentricDataset(Dataset):
    """Stage 2: lensed (0) vs eccentric (1)"""
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        arr = torch.from_numpy(np.load(file_path).astype(np.float32))
        if self.transform: arr = self.transform(arr)
        label_str = class_from_file_path(file_path)
        label = 0 if label_str == "lensed" else 1
        return arr, label, 1.0, file_path


def custom_collate_fn(batch):
    imgs, lbls, wts, paths = zip(*batch)
    return torch.stack(imgs), torch.tensor(lbls), torch.tensor(wts), paths


# -------------------------------------------------------------------------
#                            DATA LOADER
# -------------------------------------------------------------------------

def make_dataloaders(stage, path_name, seed, batch_size):
    all_files = discover_npy_files(path_name)

    if stage == 1:
        files = [f for f in all_files if any(x in f for x in ["unlensed", "lensed", "eccentric"])]
        dataset_class = UnlensedVsOthersDataset
    else:
        files = [f for f in all_files if any(x in f for x in ["lensed", "eccentric"])]
        dataset_class = LensedVsEccentricDataset

    labels = [class_from_file_path(f) for f in files]

    train_files, temp_files, train_labels, temp_labels = train_test_split(
        files, labels, test_size=0.3, stratify=labels, random_state=seed
    )
    val_files, test_files, *_ = train_test_split(
        temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=seed
    )

    transform = transforms.Compose([transforms.Resize((224, 224))])

    train_data = dataset_class(train_files, transform)
    val_data = dataset_class(val_files, transform)
    test_data = dataset_class(test_files, transform)
    num_channels = train_data[0][0].shape[0]

    num_workers = max(1, min(8, os.cpu_count() or 1))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=custom_collate_fn)

    return num_channels, train_loader, val_loader, test_loader, test_files


# -------------------------------------------------------------------------
#                            MODEL
# -------------------------------------------------------------------------

def get_model(name, num_channels, device):
    if name == "CvT":
        return CvT(num_classes=2, s1_emb_dim=64, s1_emb_kernel=7, s1_emb_stride=4,
                   s1_proj_kernel=3, s1_kv_proj_stride=2, s1_heads=1, s1_depth=1,
                   s1_mlp_mult=4, s2_emb_dim=192, s2_emb_kernel=3, s2_emb_stride=2,
                   s2_proj_kernel=3, s2_kv_proj_stride=2, s2_heads=3, s2_depth=2,
                   s2_mlp_mult=4, s3_emb_dim=384, s3_emb_kernel=3, s3_emb_stride=2,
                   s3_proj_kernel=3, s3_kv_proj_stride=2, s3_heads=4, s3_depth=10,
                   s3_mlp_mult=4, dropout=0.4).to(device)
    else:
        transformer = Linformer(dim=128, seq_len=50, depth=12, heads=8, k=64, dropout=0.4)
        return ViT(dim=128, image_size=224, patch_size=32, num_classes=2,
                   transformer=transformer, channels=num_channels).to(device)


# -------------------------------------------------------------------------
#                            TRAINING
# -------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, device, epochs, lr, gamma):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=gamma)
    criterion = nn.CrossEntropyLoss(reduction="none")

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        t_loss, t_acc = 0, 0
        for data, labels, weights, *_ in train_loader:
            data, labels, weights = data.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = (criterion(out, labels) * weights).mean()
            loss.backward()
            optimizer.step()
            acc = (out.argmax(1) == labels).float().mean()
            t_loss += loss.item(); t_acc += acc.item()

        model.eval()
        v_loss, v_acc = 0, 0
        with torch.no_grad():
            for data, labels, *_ in val_loader:
                data, labels = data.to(device), labels.to(device)
                out = model(data)
                loss = criterion(out, labels).mean()
                acc = (out.argmax(1) == labels).float().mean()
                v_loss += loss.item(); v_acc += acc.item()

        scheduler.step()
        train_losses.append(t_loss/len(train_loader))
        val_losses.append(v_loss/len(val_loader))
        train_accs.append(t_acc/len(train_loader))
        val_accs.append(v_acc/len(val_loader))
        print(f"Epoch {epoch+1}: Train {train_losses[-1]:.4f}, Val {val_losses[-1]:.4f}, "
              f"TrainAcc {train_accs[-1]:.4f}, ValAcc {val_accs[-1]:.4f}")

    return train_losses, val_losses, train_accs, val_accs


# -------------------------------------------------------------------------
#                            EVALUATION & INFERENCE
# -------------------------------------------------------------------------

def evaluate_model(model, test_loader, device, class_names, results_dir):
    fpr, tpr, roc_auc, labels, preds, *_ = compute_roc_auc_with_misclassifications(
        model, test_loader, device, class_names
    )
    plot_roc_curves(fpr, tpr, roc_auc, class_names, "", results_dir)
    plot_confusion_matrix(labels, preds, class_names, "", results_dir)
    report = classification_report(labels, preds, target_names=class_names)
    with open(results_dir / "classification_report.txt", "w") as f:
        f.write(report)
    print(report)
    return report


def hierarchical_inference(model1, model2, test_files, device):
    model1.eval(); model2.eval()
    results = {}
    for fp in test_files:
        arr = torch.from_numpy(np.load(fp).astype(np.float32)).unsqueeze(0).to(device)
        p1 = model1(arr).argmax(1).item()
        if p1 == 0:
            results[fp] = "unlensed"
        else:
            p2 = model2(arr).argmax(1).item()
            results[fp] = "lensed" if p2 == 0 else "eccentric"
    return results


# -------------------------------------------------------------------------
#                            MAIN
# -------------------------------------------------------------------------

def main():
    args = parse_args()
    seed_everything(args.seed)
    setup_directories()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n🚀 Using device: {device}")
    print(f"Model: {args.model} | Epochs: {args.epochs} | LR: {args.lr}\n")

    # ---------------- Stage 1 ----------------
    print("🧩 Training Stage 1: unlensed vs (lensed + eccentric)")
    num_channels, train_loader, val_loader, test_loader, _ = make_dataloaders(
        1, args.path_name, args.seed, args.batch_size
    )
    model1 = get_model(args.model, num_channels, device)
    if args.model_path and os.path.exists(args.model_path):
        model1.load_state_dict(torch.load(args.model_path, map_location=device))
    tL, vL, tA, vA = train_model(model1, train_loader, val_loader, device, args.epochs, args.lr, args.gamma)
    evaluate_model(model1, test_loader, device, ["unlensed", "others"], Path("./results/stage1_results"))
    plot_training_curves(tL, vL, tA, vA, Path("./results/stage1_results"))
    torch.save(model1.state_dict(), "./models/model_stage1.pth")

    # ---------------- Stage 2 ----------------
    print("\n🧩 Training Stage 2: lensed vs eccentric")
    num_channels, train_loader, val_loader, test_loader, test_files = make_dataloaders(
        2, args.path_name, args.seed, args.batch_size
    )
    model2 = get_model(args.model, num_channels, device)
    tL, vL, tA, vA = train_model(model2, train_loader, val_loader, device, args.epochs, args.lr, args.gamma)
    evaluate_model(model2, test_loader, device, ["lensed", "eccentric"], Path("./results/stage2_results"))
    plot_training_curves(tL, vL, tA, vA, Path("./results/stage2_results"))
    torch.save(model2.state_dict(), "./models/model_stage2.pth")

    # ---------------- Hierarchical Inference ----------------
    if not args.no_inference:
        print("\n⚙️ Running hierarchical inference demo...")
        results = hierarchical_inference(model1, model2, test_files[:10], device)
        for k, v in results.items():
            print(f"{k} → {v}")


if __name__ == "__main__":
    main()
