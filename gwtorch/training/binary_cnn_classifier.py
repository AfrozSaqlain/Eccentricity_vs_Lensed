import os
import glob
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
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

from gwtorch.modules.ml_utils import (
    get_model,
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
        self.file_list = file_list
        self.transform = transform
        self.df = pd.read_csv('results/' + meta_csv + '_parameters.csv')

        # set index if sample column exists
        if 'sample' in self.df.columns:
            self.df = self.df.set_index('sample')

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        # load [C, H, W] array
        arr = np.load(file_path).astype(np.float32)
        arr = torch.from_numpy(arr)

        if self.transform:
            arr = self.transform(arr)

        label_str = file_path.split("/")[-2].lower()
        label = {'lensed': 0, 'eccentric': 1}[label_str]

        file_number = int(file_path.split('/')[-1].split('.')[0])

        ln_B = 0.0
        if file_number in self.df.index:
            if label_str == 'lensed' and 'lens_lnB' in self.df.columns:
                ln_B = self.df.loc[file_number, 'lens_lnB']
            elif label_str == 'eccentric' and 'ecc_lnB' in self.df.columns:
                ln_B = self.df.loc[file_number, 'ecc_lnB']
            elif label_str == 'unlensed':
                ln_B = 100

        # weight = 10.0 if ln_B > 10 else 1.0
        weight = 1.0

        return arr, label, weight, file_path
    
def custom_collate_fn(batch):
    images, labels, weights, file_paths = zip(*batch)
    return torch.stack(images), torch.tensor(labels), torch.tensor(weights, dtype=torch.float32), file_paths

def filenames_to_label_indices(file_list):
    labels = []
    for p in file_list:
        lab = p.split('/')[-2].lower()
        labels.append({'lensed': 0, 'eccentric': 1}[lab])
    return labels

def load_all_data(path_name='data', seed=42, batch_size=32):
    
    all_files = glob.glob(f'{path_name}/eccentric/*.npy') + glob.glob(f'{path_name}/lensed/*.npy')
    all_labels = [path.split('/')[-2] for path in all_files]

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

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])

    meta_csv_path = all_files[0].split('/')[-3]
    train_data = GWDataset(train_files, transform, meta_csv=meta_csv_path)
    val_data   = GWDataset(val_files, transform, meta_csv=meta_csv_path)
    test_data  = GWDataset(test_files, transform, meta_csv=meta_csv_path)

    num_channels = train_data[0][0].shape[0] if isinstance(train_data[0][0], torch.Tensor) else 3

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=os.cpu_count(), collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                              num_workers=os.cpu_count(), collate_fn=custom_collate_fn)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                              num_workers=os.cpu_count(), collate_fn=custom_collate_fn)

    return num_channels, train_loader, val_loader, test_loader, class_weights

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

    model = get_model(name=args.model, num_channels=num_channels, device=device)

    if os.path.exists(f'./models/{args.model_file_name}.pth'):
        print("Pretrained model found. Loading...\n")
        model.load_state_dict(torch.load(f'./models/Binary_{args.model_file_name}.pth', map_location=device))
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