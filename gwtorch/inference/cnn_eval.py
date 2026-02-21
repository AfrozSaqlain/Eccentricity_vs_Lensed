import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from gwtorch.modules.general_utils import (
    compute_roc_auc_with_misclassifications,
    plot_roc_curves,
    plot_confusion_matrix,
    print_misclassified_files,
    save_misclassified_files_to_txt,
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
from gwtorch.modules.pathing import class_from_file_path


def setup_directories(results_dir, model_name='CNN'):
    os.makedirs(f'./{results_dir}/{model_name}_results/Plots', exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Test CNN model for GW classification")
    parser.add_argument('--model', type=str, default='CNN', choices=['CNN', 'SimpleCNN'], help='Model architecture to evaluate')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--model-path', type=str, default='cnn_model.pth', help='Model file name inside models directory')
    parser.add_argument('--path-name', type=str, default='data', help='Path to the dataset directory. Default is "data".')
    parser.add_argument('--results-dir', type=str, default='inference_results', help='Path to save inference results.')
    parser.add_argument('--meta-csv', type=str, default='./results/data_parameters.csv', help='Path to metadata CSV file.')
    parser.add_argument('--eval-split', type=str, default='test', choices=['test', 'all'],
                        help='Dataset subset used for evaluation. "test" follows the same 70/15/15 split as training using --seed.')
    parser.add_argument('--allow-all-eval', action='store_true',
                        help='Required with --eval-split all to avoid accidental full-dataset evaluation.')
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


def load_all_data(path_name='data', batch_size=32, meta_csv='./results/data_parameters.csv', seed=42, eval_split='test'):
    all_files = discover_npy_files(path_name)
    if not all_files:
        raise FileNotFoundError(f"No .npy files found under {path_name}/*/*.npy")
    all_labels = [class_from_file_path(path) for path in all_files]

    if eval_split == 'test':
        _, temp_files, _, temp_labels = train_test_split(
            all_files, all_labels, test_size=0.3, stratify=all_labels, random_state=seed
        )
        _, eval_files, *_ = train_test_split(
            temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=seed
        )
    else:
        eval_files = all_files

    transform = default_transform_224(antialias=False)

    test_data = GWDataset(eval_files, transform=transform, meta_csv=meta_csv)

    num_channels = test_data[0][0].shape[0]

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=safe_num_workers(),
        collate_fn=custom_collate_fn,
    )

    return num_channels, test_loader, len(eval_files), len(all_files)


def evaluate_and_plot(model, test_loader, device, results_dir, model_name='CNN'):
    class_names = ['Lensed', 'Eccentric', 'Quasi-Circular Unlensed']
    results_dir = Path(f'./{results_dir}/{model_name}_results')

    test_fpr, test_tpr, test_roc_auc, test_labels, test_preds, _, test_misclassified = compute_roc_auc_with_misclassifications(
        model, test_loader, device, class_names
    )

    print_misclassified_files(test_misclassified, "Test Set")
    save_misclassified_files_to_txt(test_misclassified, "Test", results_dir / "misclassified_test.txt")

    plot_roc_curves(test_fpr, test_tpr, test_roc_auc, class_names, " (Test Set)", results_dir)
    plot_confusion_matrix(test_labels, test_preds, class_names, " (Test Set)", results_dir)

    print("\nTest Classification Report:")
    test_report = classification_report(test_labels, test_preds, target_names=class_names)
    print(test_report)
    test_report_path = results_dir / "test_classification_report.txt"
    with open(test_report_path, "w") as f:
        f.write("Test Classification Report:\n")
        f.write(test_report)

    return test_misclassified


def main():
    args = parse_args()
    if args.eval_split == 'all' and not args.allow_all_eval:
        raise ValueError("Using --eval-split all requires --allow-all-eval.")

    setup_directories(results_dir=args.results_dir, model_name=args.model)
    seed_everything(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    num_channels, test_loader, eval_count, total_count = load_all_data(
        path_name=args.path_name,
        batch_size=args.batch_size,
        meta_csv=args.meta_csv,
        seed=args.seed,
        eval_split=args.eval_split,
    )
    if args.eval_split == 'test':
        print(f"Evaluating held-out test split: {eval_count}/{total_count} samples")
    else:
        print(f"Evaluating all samples (explicit opt-in): {eval_count}/{total_count} samples")

    model = get_model(name=args.model, num_channels=num_channels, device=device)

    model_file = Path('./models') / args.model_path
    if model_file.exists():
        model.load_state_dict(torch.load(model_file, map_location=device))
        print(f"Model loaded from {model_file}")
    else:
        raise FileNotFoundError(f"No pre-trained model found at {model_file}.")

    test_misclassified = evaluate_and_plot(model, test_loader, device, args.results_dir, model_name=args.model)

    print("\nInference complete. Summary:")
    print(f"Test misclassifications: {len(test_misclassified)}")


if __name__ == '__main__':
    main()
