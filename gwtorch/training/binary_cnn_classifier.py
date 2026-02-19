import os
import random
import argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from gwtorch.modules.general_utils import (
    compute_roc_auc_with_misclassifications,
    plot_roc_curves,
    plot_confusion_matrix,
    plot_training_curves,
    print_misclassified_files,
    save_misclassified_files_to_txt,
    plot_training_curves,
)

from gwtorch.modules.neural_net import CNNBinaryModel, SimpleCNNBinary
from gwtorch.modules.data_loading import (
    NpyGWDataset,
    collate_with_weights,
    default_transform_224,
    safe_num_workers,
)
from gwtorch.modules.pathing import (
    class_from_file_path,
    default_meta_csv_path,
    discover_npy_files
)


class_names = ['lensed', 'eccentric']

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def setup_directories(model_name):
    os.makedirs('./models', exist_ok=True)
    os.makedirs(f'./results/Binary_{model_name}_results/Plots', exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN for binary GW classification (eccentric vs lensed)")
    parser.add_argument('--model', type=str, default='CNN', choices=['CNN', 'SimpleCNN'], help='Model to train')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-file-name', type=str, default='cnn_model_binary')
    parser.add_argument('--path-name', type=str, default='data', help='Path to the dataset directory. Default is "data".')
    return parser.parse_args()

class GWDataset(Dataset):
    def __init__(self, file_list, transform=None, meta_csv='data'):
        meta_csv_str = str(meta_csv)
        if Path(meta_csv_str).suffix != ".csv":
            meta_csv_str = str(Path("results") / f"{meta_csv}_parameters.csv")
        self.dataset = NpyGWDataset(
            file_list=file_list,
            label_map={'lensed': 0, 'eccentric': 1},
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

def filenames_to_label_indices(file_list):
    labels = []
    for p in file_list:
        lab = class_from_file_path(p)
        labels.append({'lensed': 0, 'eccentric': 1}[lab])
    return labels

def load_all_data(path_name='data', seed=42, batch_size=32):
    all_files = [
        p for p in discover_npy_files(path_name)
        if class_from_file_path(p) in {"eccentric", "lensed"}
    ]
    if not all_files:
        raise FileNotFoundError(f"No eccentric/lensed .npy files found under {path_name}")
    all_labels = [class_from_file_path(path) for path in all_files]

    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels, test_size=0.3, stratify=all_labels, random_state=seed
    )

    val_files, test_files, *_ = train_test_split(
        temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=seed
    )

    train_labels = filenames_to_label_indices(train_files)
    counts = Counter(train_labels)
    total = sum(counts.values())
    class_weights = torch.tensor([ total/counts[i] for i in range(2) ], dtype=torch.float32)

    transform = default_transform_224(antialias=False)

    meta_csv_path = default_meta_csv_path(path_name)
    train_data = GWDataset(train_files, transform, meta_csv=meta_csv_path)
    val_data   = GWDataset(val_files, transform, meta_csv=meta_csv_path)
    test_data  = GWDataset(test_files, transform, meta_csv=meta_csv_path)

    num_channels = train_data[0][0].shape[0] if isinstance(train_data[0][0], torch.Tensor) else 3

    num_workers = safe_num_workers()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=custom_collate_fn)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=custom_collate_fn)

    return num_channels, train_loader, val_loader, test_loader, class_weights


def get_binary_model(name, num_channels, device):
    if name == 'CNN':
        return CNNBinaryModel(input_channels=num_channels, dropout=0.2).to(device)
    return SimpleCNNBinary(input_channels=num_channels, dropout=0.2).to(device)

def train_model(model, train_loader, valid_loader, device, epochs, lr, gamma, class_weights=None):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=gamma)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(reduction='none', weight=class_weights.to(device), label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss, running_acc = 0, 0
        for data, labels, weights, *_ in train_loader:
            data, labels, weights = data.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad()
            output = model(data)

            unweighted_loss = criterion(output, labels)
            weighted_loss = unweighted_loss * weights
            loss = weighted_loss.mean()
            
            loss.backward()
            optimizer.step()
            acc = (output.argmax(1) == labels).float().mean()
            running_loss += loss.item()
            running_acc += acc.item()

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(running_acc / len(train_loader))

        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for data, labels, *_ in valid_loader:
                data, labels = data.to(device), labels.to(device)
                output = model(data)
                loss = criterion(output, labels)
                loss = loss.mean()
                acc = (output.argmax(1) == labels).float().mean()
                val_loss += loss.item()
                val_acc += acc.item()

        val_losses.append(val_loss / len(valid_loader))
        val_accs.append(val_acc / len(valid_loader))

        scheduler.step()

        print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Acc={train_accs[-1]:.4f}; Val Loss={val_losses[-1]:.4f}, Acc={val_accs[-1]:.4f}")

    return train_losses, val_losses, train_accs, val_accs

def evaluate_and_plot(model, valid_loader, test_loader, device, fold=None, model_name=None):
    """Evaluate model on validation and test sets, plot results, and save misclassified files"""
    results_dir = Path(f'./results/Binary_{model_name}_results')
    
    if fold is not None:
        fold_dir = results_dir / f'fold_{fold}'
        fold_dir.mkdir(exist_ok=True)
        results_dir = fold_dir

    val_fpr, val_tpr, val_roc_auc, val_labels, val_preds, _, val_misclassified = compute_roc_auc_with_misclassifications(model, valid_loader, device, class_names)
    test_fpr, test_tpr, test_roc_auc, test_labels, test_preds, _, test_misclassified = compute_roc_auc_with_misclassifications(model, test_loader, device, class_names)

    if fold is not None:
        print(f"\n=== FOLD {fold} RESULTS ===")
    
    print_misclassified_files(val_misclassified, "Validation Set")
    print_misclassified_files(test_misclassified, "Test Set")
    save_misclassified_files_to_txt(val_misclassified, "Validation", results_dir / "misclassified_validation.txt")
    save_misclassified_files_to_txt(test_misclassified, "Test", results_dir / "misclassified_test.txt")

    suffix = f" Fold {fold}" if fold is not None else ""


    plot_roc_curves(val_fpr, val_tpr, val_roc_auc, class_names, f" (Validation Set{suffix})", results_dir)
    plot_roc_curves(test_fpr, test_tpr, test_roc_auc, class_names, f" (Test Set{suffix})", results_dir)
    plot_confusion_matrix(val_labels, val_preds, class_names, f" (Validation Set{suffix})", results_dir)
    plot_confusion_matrix(test_labels, test_preds, class_names, f" (Test Set{suffix})", results_dir)

    print("Validation Classification Report:")
    val_report = classification_report(val_labels, val_preds, target_names=class_names)
    print(val_report)
    val_report_path = results_dir / "val_classification_report.txt"
    with open(val_report_path, "w") as f:
        f.write("Validation Classification Report:\n")
        f.write(val_report)

    print("\nTest Classification Report:")
    test_report = classification_report(test_labels, test_preds, target_names=class_names)
    print(test_report)
    test_report_path = results_dir / "test_classification_report.txt"
    with open(test_report_path, "w") as f:
        f.write("Test Classification Report:\n")
        f.write(test_report)

    return val_misclassified, test_misclassified, val_roc_auc, test_roc_auc

# ----------------- Main -----------------
def main():
    args = parse_args()
    setup_directories(model_name=args.model)
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    print("Training binary classifier: Eccentric vs Lensed")
    
    num_channels, train_loader, valid_loader, test_loader, class_weights = load_all_data(path_name=args.path_name, seed=args.seed, batch_size=args.batch_size)

    model = get_binary_model(name=args.model, num_channels=num_channels, device=device)

    model_path = Path(f'./models/Binary_{args.model_file_name}.pth')
    if model_path.exists():
        print("Pretrained model found. Loading...\n")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("No pretrained model found. Training from scratch...\n")

    if torch.all(class_weights == class_weights[0]):
        class_weights = None
        print(f"Found all class samples to be equal. Setting class weights to {class_weights}")

    train_losses, val_losses, train_accs, val_accs = train_model(
        model=model, train_loader=train_loader, valid_loader=valid_loader, device=device, epochs=args.epochs, lr=args.lr, gamma=args.gamma, class_weights=class_weights
    )
    
    val_misclassified, test_misclassified, _, _ = evaluate_and_plot(
        model=model, valid_loader=valid_loader, test_loader=test_loader, device=device , model_name=args.model
    )

    plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                        Path(f'./results/Binary_{args.model}_results'))

    print("\nTraining complete. Summary:")
    print(f"Validation misclassifications: {len(val_misclassified)}")
    print(f"Test misclassifications: {len(test_misclassified)}")

    torch.save(model.state_dict(), f'./models/Binary_{args.model_file_name}.pth')
    print(f"Model saved to ./models/Binary_{args.model_file_name}.pth")

if __name__ == "__main__":
    main()
