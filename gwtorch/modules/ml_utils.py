import os
import glob
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold

from gwtorch.modules.neural_net import (
    CNN_Model,
    SimpleCNN,
    create_cvt_model,
    create_vit_model,
    create_vit_linear_model,
)

from gwtorch.modules.general_utils import (
    compute_roc_auc_with_misclassifications,
    plot_roc_curves,
    plot_confusion_matrix,
    print_misclassified_files,
    save_misclassified_files_to_txt,
    plot_training_curves,
    plot_kfold_roc_curves,
    plot_average_training_curves
)
###################################################################################################
# Constants for ML training and evaluation. Adjust as needed for different experiments or datasets.
###################################################################################################

dropout = 0.2
class_names = ["Lensed", "Eccentric", "Quasi-Circular Unlensed"]

###################################################################################################

def _collect_dataset_files(path_name):
    all_files = glob.glob(f'{path_name}/*/*.npy')
    if not all_files:
        raise FileNotFoundError(
            f"No '.npy' files found under '{path_name}/*/'. "
            "Expected class subdirectories (e.g., lensed/eccentric/unlensed)."
        )
    return all_files

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_directories(model_name=None):
    os.makedirs('./models', exist_ok=True)
    os.makedirs(f'./results/{model_name}_results/Plots', exist_ok=True)
    
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
        label = {'lensed': 0, 'eccentric': 1, 'unlensed': 2}[label_str]

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
        labels.append({'lensed':0,'eccentric':1,'unlensed':2}[lab])
    return labels

def load_all_data(path_name='data', seed=42, batch_size=32):

    all_files = _collect_dataset_files(path_name)
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
    class_weights = torch.tensor([ total/counts[i] for i in range(3) ], dtype=torch.float32)

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

def prepare_kfold_data(path_name='data'):
    """Prepare data for k-fold cross validation"""

    all_files = _collect_dataset_files(path_name)
    all_labels = [path.split('/')[-2] for path in all_files]
    
    return all_files, all_labels

def create_fold_loaders(train_files, val_files, batch_size):
    """Create data loaders for a specific fold"""

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])

    meta_csv_path = train_files[0].split('/')[-3]

    train_data = GWDataset(train_files, transform, meta_csv=meta_csv_path)
    val_data   = GWDataset(val_files, transform, meta_csv=meta_csv_path)

    num_channels = train_data[0][0].shape[0] if isinstance(train_data[0][0], torch.Tensor) else 3

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=os.cpu_count(), collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                              num_workers=os.cpu_count(), collate_fn=custom_collate_fn)

    return num_channels, train_loader, val_loader

def get_model(name, num_channels, device, num_classes=3, dropout_rate=dropout):
    if name == 'CvT':
        model = create_cvt_model(num_classes=num_classes, num_channels=num_channels, dropout=dropout_rate)
    elif name == 'ViT':
        model = create_vit_model(num_classes=num_classes, num_channels=num_channels, dropout=dropout_rate)
    elif name == 'ViTLinear':
        model = create_vit_linear_model(num_classes=num_classes, num_channels=num_channels, dropout=dropout_rate)
    elif name == 'CNN':
        model = CNN_Model(input_channels=num_channels, dropout=dropout_rate, num_classes=num_classes)
    elif name == 'SimpleCNN':
        model = SimpleCNN(input_channels=num_channels, dropout=dropout_rate, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")

    return model.to(device)

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

def evaluate_model(model, data_loader, device):
    """Evaluate model and return predictions and labels"""
    model.eval()
    all_preds = []
    all_labels = []
    all_file_paths = []
    
    with torch.no_grad():
        for data, labels, _, file_paths in data_loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            preds = output.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_file_paths.extend(file_paths)
    
    return np.array(all_preds), np.array(all_labels), all_file_paths

def evaluate_and_plot(model, valid_loader, test_loader, device, fold=None, model_name=None):
    """Evaluate model on validation and test sets, plot results, and save misclassified files"""
    results_dir = Path(f'results/{model_name}_results')
    
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

def run_kfold_training(args, device, get_model=None, model_name=None):
    """Run k-fold cross validation training"""
    print(f"\n{'='*50}")
    print(f"Starting {args.k_folds}-Fold Cross Validation")
    print(f"{'='*50}")

    model = get_model(name=args.model, num_channels=4, device=device)

    all_files, all_labels = prepare_kfold_data(path_name=args.path_name)
    
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    fold_results = {
        'val_accuracies': [],
        'test_accuracies': [],  # We'll use validation as test for k-fold
        'val_roc_aucs': [],
        'train_losses_all': [],
        'val_losses_all': [],
        'train_accs_all': [],
        'val_accs_all': [],
        'roc_data': []  # Store FPR, TPR, AUC for each fold
    }

    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        all_files, all_labels, test_size=0.2, stratify=all_labels, random_state=args.seed
    )

    transform = transforms.Compose([
        # transforms.Lambda(lambda x: torch.tensor(x).permute(2, 0, 1)),
        transforms.Resize((224, 224)),
    ])

    meta_csv_path = all_files[0].split('/')[-3]
    
    test_data = GWDataset(test_files, transform, meta_csv=meta_csv_path)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                           num_workers=os.cpu_count(), collate_fn=custom_collate_fn)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_files, train_val_labels)):
        
        train_files_fold = [train_val_files[i] for i in train_idx]
        val_files_fold = [train_val_files[i] for i in val_idx]

        train_labels_fold = filenames_to_label_indices(train_files_fold)
        counts = Counter(train_labels_fold)
        total = sum(counts.values())
        class_weights = torch.tensor([ total/counts[i] for i in range(3) ], dtype=torch.float32)

        num_channels, train_loader, val_loader = create_fold_loaders(train_files_fold, val_files_fold, args.batch_size)
        
        print(f"\n{'='*30}")
        print(f"FOLD {fold + 1}/{args.k_folds}")
        print(f"{'='*30}")

        model = get_model(name=args.model, num_channels=num_channels, device=device)
        
        print(f"Training fold {fold + 1}...")
        print(f"Train samples: {len(train_files_fold)}, Val samples: {len(val_files_fold)}")

        train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, device, args.epochs, args.lr, args.gamma, class_weights=class_weights
        )

        fold_results['train_losses_all'].append(train_losses)
        fold_results['val_losses_all'].append(val_losses)
        fold_results['train_accs_all'].append(train_accs)
        fold_results['val_accs_all'].append(val_accs)

        val_misclassified, test_misclassified, val_roc_auc, _test_roc_auc = evaluate_and_plot(
            model, val_loader, test_loader, device, fold=fold+1, model_name=model_name
        )
        test_preds, test_true, _ = evaluate_model(model, test_loader, device)
        test_accuracy = float(np.mean(test_preds == test_true))

        val_fpr, val_tpr, val_roc_auc_detailed, _, _, _, _ = compute_roc_auc_with_misclassifications(
            model, val_loader, device, class_names
        )

        fold_results['roc_data'].append((val_fpr, val_tpr, val_roc_auc_detailed))

        fold_results['val_accuracies'].append(val_accs[-1])
        fold_results['test_accuracies'].append(test_accuracy)
        fold_results['val_roc_aucs'].append(val_roc_auc)

        fold_dir = Path(f'results/{model_name}_results') / f'fold_{fold+1}'
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, fold_dir)

        model_path = f'models/{args.model_file_name.split(".")[0]}_fold_{fold+1}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model for fold {fold+1} saved to {model_path}")
        
        print(f"Fold {fold+1} - Final Val Accuracy: {val_accs[-1]:.4f}")
        print(f"Fold {fold+1} - Val misclassifications: {len(val_misclassified)}")
        print(f"Fold {fold+1} - Test misclassifications: {len(test_misclassified)}")

    print(f"\n{'='*50}")
    print("K-FOLD CROSS VALIDATION SUMMARY")
    print(f"{'='*50}")
    
    val_acc_mean = np.mean(fold_results['val_accuracies'])
    val_acc_std = np.std(fold_results['val_accuracies'])
    
    print(f"Validation Accuracy: {val_acc_mean:.4f} ± {val_acc_std:.4f}")
    print(f"Individual fold accuracies: {[f'{acc:.4f}' for acc in fold_results['val_accuracies']]}")

    # Populates mean/std AUC keys inside fold_results for summary output.
    plot_kfold_roc_curves(fold_results, Path(f'results/{model_name}_results'), class_names)

    summary_path = Path(f'./results/{model_name}_results/kfold_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"K-Fold Cross Validation Summary ({args.k_folds} folds)\n")
        f.write(f"{'='*50}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Seed: {args.seed}\n\n")
        f.write(f"Validation Accuracy: {val_acc_mean:.4f} ± {val_acc_std:.4f}\n")
        f.write(f"Individual fold accuracies: {fold_results['val_accuracies']}\n\n")

        if 'mean_auc_lensed' in fold_results:
            f.write("ROC AUC Summary:\n")
            f.write("-" * 20 + "\n")
            for class_name in class_names:
                class_key = class_name.lower().replace(" ", "_").replace("-", "_")
                if f'mean_auc_{class_key}' in fold_results:
                    mean_auc = fold_results[f'mean_auc_{class_key}']
                    std_auc = fold_results[f'std_auc_{class_key}']
                    f.write(f"{class_name}: {mean_auc:.4f} ± {std_auc:.4f}\n")

    plot_average_training_curves(fold_results, Path(f'results/{model_name}_results'))
    
    return fold_results
