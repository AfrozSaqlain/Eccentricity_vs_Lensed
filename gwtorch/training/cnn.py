import os
import glob
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

import random
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold

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

from gwtorch.modules.neural_net import CNN_Model

# ----------------- Constants -----------------
RANDOM_SEED = 42
BATCH_SIZE = 512
EPOCHS = 20
LR = 3e-5
GAMMA = 0.7
K_FOLDS = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_names = ["Lensed", "Eccentric", "Quasi-Circular Unlensed"]

# ----------------- Functions -----------------
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def setup_directories():
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./results/cnn_results/Plots', exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN model for GW classification")
    parser.add_argument('--model', type=str, default='CNN', choices=['CNN'], help='Model to train')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--gamma', type=float, default=GAMMA)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--model-path', type=str, default='cnn_model.pth')
    # parser.add_argument('--num-channels', type=int, default=3, help="Defines what is the input channels of the model. Default is 3 for signals from 3 detectors.")
    parser.add_argument('--k-folds', type=int, default=K_FOLDS, help="Number of folds for k-fold cross validation")
    parser.add_argument('--use-kfold', action='store_true', help="Use k-fold cross validation instead of single train/val/test split")
    parser.add_argument('--path-name', type=str, default='data', help='Path to the dataset directory. Default is "data".')
    return parser.parse_args()

# ----------------- Dataset Class -----------------
class GWDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        arr = np.load(file_path)  # shape: (H, W, C)

        # Transpose to (H, W, C) only if needed
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]  # Convert to (H, W, 1)

        arr = arr.astype(np.float32)
        
        arr = self.transform(arr)

        label_str = file_path.split("/")[-1].split("_")[0]
        label = {'lensed': 0, 'eccentric': 1, 'unlensed': 2}[label_str]
        file_number = file_path.split("/")[-1].split("_")[1].split(".")[0]

        return arr, label, file_number, file_path

# ----------------- Utility Functions -----------------
def custom_collate_fn(batch):
    images, labels, file_numbers, img_paths = zip(*batch)
    return torch.stack(images), torch.tensor(labels), file_numbers, img_paths

def load_all_data(path_name='data', seed=42, batch_size=42):

    all_files = glob.glob(f'./{path_name}/*.npy')
    all_labels = [path.split('/')[-1].split('_')[0] for path in all_files]

    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels, test_size=0.3, stratify=all_labels, random_state=seed
    )

    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=seed
    )

    transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x).permute(2, 0, 1)),
        transforms.Resize((224, 224)),
    ])
    
    train_data = GWDataset(train_files, transform)
    val_data   = GWDataset(val_files, transform)
    test_data  = GWDataset(test_files, transform)

    num_channels = train_data[0][0].shape[0] if isinstance(train_data[0][0], torch.Tensor) else 3

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=os.cpu_count(), collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                              num_workers=os.cpu_count(), collate_fn=custom_collate_fn)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                              num_workers=os.cpu_count(), collate_fn=custom_collate_fn)

    return num_channels, train_loader, val_loader, test_loader

def prepare_kfold_data(path_name='data'):
    """Prepare data for k-fold cross validation"""

    all_files = glob.glob(f'./{path_name}/*.npy')
    all_labels = [path.split('/')[-1].split('_')[0] for path in all_files]
    
    return all_files, all_labels

def create_fold_loaders(train_files, val_files, batch_size):
    """Create data loaders for a specific fold"""

    transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x).permute(2, 0, 1)),
        transforms.Resize((224, 224)),
    ])

    train_data = GWDataset(train_files, transform)
    val_data   = GWDataset(val_files, transform)

    num_channels = train_data[0][0].shape[0] if isinstance(train_data[0][0], torch.Tensor) else 3

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=os.cpu_count(), collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                              num_workers=os.cpu_count(), collate_fn=custom_collate_fn)

    return num_channels, train_loader, val_loader

def train_model(model, train_loader, valid_loader, device, epochs, lr, gamma):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss, running_acc = 0, 0
        for data, labels, *_ in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
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
        for data, labels, file_numbers, file_paths in data_loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            preds = output.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_file_paths.extend(file_paths)
    
    return np.array(all_preds), np.array(all_labels), all_file_paths

def evaluate_and_plot(model, valid_loader, test_loader, device, fold=None):
    """Evaluate model on validation and test sets, plot results, and save misclassified files"""
    results_dir = Path('./results/cnn_results')
    
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

def run_kfold_training(args, device):
    """Run k-fold cross validation training"""
    print(f"\n{'='*50}")
    print(f"Starting {args.k_folds}-Fold Cross Validation")
    print(f"{'='*50}")
    
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
        transforms.Lambda(lambda x: torch.tensor(x).permute(2, 0, 1)),
        transforms.Resize((224, 224)),
    ])
    test_data = GWDataset(test_files, transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                           num_workers=os.cpu_count(), collate_fn=custom_collate_fn)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_files, train_val_labels)):
        
        train_files_fold = [train_val_files[i] for i in train_idx]
        val_files_fold = [train_val_files[i] for i in val_idx]

        num_channels, train_loader, val_loader = create_fold_loaders(train_files_fold, val_files_fold, args.batch_size)

        if fold == 0:
            print(f"\nNumber of channels in data: {num_channels}")
        
        print(f"\n{'='*30}")
        print(f"FOLD {fold + 1}/{args.k_folds}")
        print(f"{'='*30}")


        model = CNN_Model(input_channlels=num_channels).to(device)
        
        print(f"Training fold {fold + 1}...")
        print(f"Train samples: {len(train_files_fold)}, Val samples: {len(val_files_fold)}")

        train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, device, args.epochs, args.lr, args.gamma
        )

        fold_results['train_losses_all'].append(train_losses)
        fold_results['val_losses_all'].append(val_losses)
        fold_results['train_accs_all'].append(train_accs)
        fold_results['val_accs_all'].append(val_accs)

        val_misclassified, test_misclassified, val_roc_auc, test_roc_auc = evaluate_and_plot(
            model, val_loader, test_loader, device, fold=fold+1
        )

        val_fpr, val_tpr, val_roc_auc_detailed, _, _, _, _ = compute_roc_auc_with_misclassifications(
            model, val_loader, device, class_names
        )

        fold_results['roc_data'].append((val_fpr, val_tpr, val_roc_auc_detailed))

        fold_results['val_accuracies'].append(val_accs[-1])
        fold_results['test_accuracies'].append(test_roc_auc) 
        fold_results['val_roc_aucs'].append(val_roc_auc)

        fold_dir = Path('./results/cnn_results') / f'fold_{fold+1}'
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, fold_dir)

        model_path = f'./models/{args.model_path.split(".")[0]}_fold_{fold+1}.pth'
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

    summary_path = Path('./results/cnn_results/kfold_summary.txt')
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

        if hasattr(fold_results, 'mean_auc_lensed'):
            f.write("ROC AUC Summary:\n")
            f.write("-" * 20 + "\n")
            for class_name in class_names:
                class_key = class_name.lower().replace(" ", "_").replace("-", "_")
                if f'mean_auc_{class_key}' in fold_results:
                    mean_auc = fold_results[f'mean_auc_{class_key}']
                    std_auc = fold_results[f'std_auc_{class_key}']
                    f.write(f"{class_name}: {mean_auc:.4f} ± {std_auc:.4f}\n")

    plot_average_training_curves(fold_results, Path('./results/cnn_results'))

    plot_kfold_roc_curves(fold_results, Path('./results/cnn_results'), class_names)
    
    return fold_results

# ----------------- Main -----------------
def main():
    args = parse_args()
    setup_directories()
    seed_everything(args.seed)

    results_dir = Path('./results/cnn_results')

    print(f"Using device: {device}")

    if args.use_kfold:
        fold_results = run_kfold_training(args, device)
        print(f"\nK-fold cross validation completed!")
    else:
        print("\nRunning single train/validation/test split...")
        num_channels, train_loader, valid_loader, test_loader = load_all_data(path_name=args.path_name, seed=args.seed, batch_size=args.batch_size)

        model = CNN_Model(input_channlels=num_channels).to(device)

        print(f"\nNumber of channels in data: {num_channels}")
        
        if os.path.exists(f'./models/{args.model_path}'):
            print("Pretrained model found. Loading...\n")
            model.load_state_dict(torch.load(f'./models/{args.model_path}'))
        else:
            print("No pretrained model found. Training from scratch...\n")

        train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, valid_loader, device, args.epochs, args.lr, args.gamma
        )
        val_misclassified, test_misclassified, _, _ = evaluate_and_plot(
            model, valid_loader, test_loader, device
        )
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                           Path('./results/cnn_results'))

        print("\nTraining complete. Summary:")
        print(f"Validation misclassifications: {len(val_misclassified)}")
        print(f"Test misclassifications: {len(test_misclassified)}")

        torch.save(model.state_dict(), f'./models/{args.model_path}')
        print(f"Model saved to ./models/{args.model_path}")

if __name__ == "__main__":
    main()
