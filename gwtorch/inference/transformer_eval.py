import os
import argparse
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

from gwtorch.modules.general_utils import (
    print_misclassified_files,
    save_misclassified_files_to_txt,
    get_file_prefix,
    plot_roc_curves,
)

from gwtorch.modules.ml_utils import (
    seed_everything,
    get_model,
)
from gwtorch.modules.data_loading import (
    NpyGWDataset,
    collate_with_lnb,
    default_transform_224,
    safe_num_workers,
)
from gwtorch.modules.pathing import discover_npy_files


def setup_directories(results_dir, model_name='ViT'):
    os.makedirs(f'./{results_dir}/{model_name}_results/Plots', exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Test Transformer for GW classification")
    parser.add_argument('--model', type=str, default='ViT', choices=['CvT', 'ViT', 'ViTLinear'], help='Model architecture to evaluate')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--model-path', type=str, default='vit_model.pth', help='Model file name inside models directory')
    parser.add_argument('--path-name', type=str, default='data', help='Path to the dataset directory.')
    parser.add_argument('--results-dir', type=str, default='inference_results', help='Path to save inference results.')
    parser.add_argument('--meta-csv', type=str, default='./results/data_parameters.csv', help='Path to the metadata CSV file.')
    parser.add_argument('--lnB-min', type=float, default=None, help='Minimum lnB value for confusion-matrix filtering.')
    parser.add_argument('--lnB-max', type=float, default=None, help='Maximum lnB value for confusion-matrix filtering.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    return parser.parse_args()


class GWDataset(Dataset):
    def __init__(self, file_list, transform=None, meta_csv='./results/data_parameters.csv'):
        self.dataset = NpyGWDataset(
            file_list=file_list,
            label_map={'lensed': 0, 'eccentric': 1, 'unlensed': 2},
            transform=transform,
            meta_csv=meta_csv,
            return_lnB=True,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def custom_collate_fn(batch):
    return collate_with_lnb(batch)


def load_all_data(path_name='data', batch_size=32, meta_csv='./results/data_parameters.csv'):
    all_files = discover_npy_files(path_name)
    if not all_files:
        raise FileNotFoundError(f"No .npy files found under {path_name}/*/*.npy")

    transform = default_transform_224(antialias=True)

    test_data = GWDataset(all_files, transform=transform, meta_csv=meta_csv)
    num_channels = test_data[0][0].shape[0]

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=safe_num_workers(),
        collate_fn=custom_collate_fn,
    )

    return num_channels, test_loader


def compute_roc_auc_with_misclassifications(model, data_loader, device, class_names):
    model.eval()
    all_labels, all_predictions, all_probabilities, all_img_paths, all_lnBs = [], [], [], [], []

    with torch.no_grad():
        for data, labels, lnBs, img_paths in data_loader:
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_img_paths.extend(img_paths)
            all_lnBs.extend(lnBs)

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_lnBs = np.array(all_lnBs)

    misclassified_mask = all_labels != all_predictions
    misclassified_indices = np.where(misclassified_mask)[0]
    misclassified_info = []
    for idx in misclassified_indices:
        misclassified_info.append({
            'file_path': all_img_paths[idx],
            'true_label': all_labels[idx],
            'predicted_label': all_predictions[idx],
            'true_class': class_names[all_labels[idx]],
            'predicted_class': class_names[all_predictions[idx]],
            'prediction_probabilities': all_probabilities[idx],
        })

    label_binarizer = LabelBinarizer()
    y_true_binary = label_binarizer.fit_transform(all_labels)
    if y_true_binary.shape[1] == 1:
        y_true_binary = np.hstack([1 - y_true_binary, y_true_binary])

    n_classes = y_true_binary.shape[1]

    fpr, tpr, roc_auc_scores = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], all_probabilities[:, i])
        roc_auc_scores[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc_scores['macro'] = auc(fpr['macro'], tpr['macro'])

    return fpr, tpr, roc_auc_scores, all_labels, all_predictions, misclassified_info, all_lnBs


def plot_confusion_matrix_percent(y_true, y_pred, class_names, title_suffix='', results_dir=None):
    if len(y_true) == 0:
        print(f"Skipping confusion matrix for '{title_suffix}': no data points after filtering.")
        return

    prefix = get_file_prefix(results_dir) if results_dir else ''
    title = title_suffix.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')

    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_percentage = np.divide(cm.astype(float) * 100, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    annotations = [[f"{cm_percentage[i, j]:.1f}%" for j in range(cm.shape[1])] for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=annotations, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f'Confusion Matrix{title_suffix}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()

    if results_dir:
        results_dir = Path(results_dir)
        os.makedirs(results_dir / 'Plots', exist_ok=True)
        fig.savefig(results_dir / f'Plots/{prefix}_Confusion_Matrix{title}.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_and_plot(model, test_loader, device, results_dir, model_name, args):
    class_names = ['Lensed', 'Eccentric', 'Quasi-Circular Unlensed']
    results_dir = Path(f'./{results_dir}/{model_name}_results')

    test_fpr, test_tpr, test_roc_auc, test_labels, test_preds, test_misclassified, test_lnBs = compute_roc_auc_with_misclassifications(
        model, test_loader, device, class_names
    )

    print_misclassified_files(test_misclassified, 'Test Set')
    save_misclassified_files_to_txt(test_misclassified, 'Test', results_dir / 'misclassified_test.txt')

    plot_roc_curves(test_fpr, test_tpr, test_roc_auc, class_names, ' (Test Set)', results_dir)
    plot_confusion_matrix_percent(test_labels, test_preds, class_names, ' (Test Set, All Data)', results_dir)

    if args.lnB_min is not None or args.lnB_max is not None:
        mask = np.ones_like(test_lnBs, dtype=bool)
        filter_parts = []

        if args.lnB_min is not None:
            mask &= (test_lnBs >= args.lnB_min)
            filter_parts.append(f'lnB >= {args.lnB_min}')

        if args.lnB_max is not None:
            mask &= (test_lnBs <= args.lnB_max)
            filter_parts.append(f'lnB <= {args.lnB_max}')

        filtered_labels = test_labels[mask]
        filtered_preds = test_preds[mask]

        filter_str = ' and '.join(filter_parts)
        title_suffix = f' (Test Set, Filtered by {filter_str})'

        print('-' * 50)
        print(f'Plotting filtered confusion matrix for: {filter_str}')
        print(f'{len(filtered_labels)} out of {len(test_labels)} samples remaining.')
        print('-' * 50)

        plot_confusion_matrix_percent(filtered_labels, filtered_preds, class_names, title_suffix, results_dir)

    print('\nTest Classification Report (on all test data):')
    report = classification_report(test_labels, test_preds, target_names=class_names, zero_division=0)
    print(report)
    with open(results_dir / 'test_classification_report.txt', 'w') as f:
        f.write('Test Classification Report:\n')
        f.write(report)


def main():
    args = parse_args()
    setup_directories(results_dir=args.results_dir, model_name=args.model)
    seed_everything(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    num_channels, test_loader = load_all_data(
        path_name=args.path_name,
        batch_size=args.batch_size,
        meta_csv=args.meta_csv,
    )

    model = get_model(name=args.model, num_channels=num_channels, device=device)

    model_file = Path('./models') / args.model_path
    if model_file.exists():
        model.load_state_dict(torch.load(model_file, map_location=device))
        print(f'Model loaded from {model_file}')
    else:
        raise FileNotFoundError(f'No pre-trained model found at {model_file}.')

    evaluate_and_plot(model, test_loader, device, args.results_dir, args.model, args)


if __name__ == '__main__':
    main()
