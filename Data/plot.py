import os
import pickle

from general_utils import (
    plot_roc_curves,
    plot_confusion_matrix,
    plot_training_curves,
    plot_kfold_roc_curves,
    plot_average_training_curves
)


def _load_pickle(path):
    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        return None
    with open(path, "rb") as file:
        return pickle.load(file)


def _plot_confusion_matrix(data_dir, model_name, results_dir=None):
    path = f"{data_dir}/{model_name}_Confusion_Matrix_Test.pkl"
    f = _load_pickle(path)
    if f is None:
        return

    if len(f['class_names']) > 2:
        f['class_names'][2] = 'Unlensed'

    plot_confusion_matrix(
        f['y_true'],
        f['y_pred'],
        class_names=f['class_names'],
        title_suffix=f['title_suffix'],
        results_dir=results_dir if results_dir else './'
    )


def _plot_roc_curves(data_dir, model_name, results_dir=None):
    path = f"{data_dir}/{model_name}_ROC_Curves_Test.pkl"
    f = _load_pickle(path)
    if f is None:
        return

    if len(f['class_names']) > 2:
        f['class_names'][2] = 'Unlensed'

    plot_roc_curves(
        fpr=f['fpr'],
        tpr=f['tpr'],
        roc_auc=f['roc_auc'],
        class_names=f['class_names'],
        title_suffix=f['title_suffix'],
        scale=f['scale'],
        results_dir=results_dir if results_dir else './'
    )


def _plot_training_curves(data_dir, model_name, results_dir=None):
    path = f"{data_dir}/{model_name}_Training_curves.pkl"
    f = _load_pickle(path)
    if f is None:
        return

    plot_training_curves(
        train_losses=f['train_losses'],
        val_losses=f['val_losses'],
        train_accuracies=f['train_accuracies'],
        val_accuracies=f['val_accuracies'],
        results_dir=results_dir if results_dir else './'
    )


def _plot_kfold_roc_curves(data_dir, results_dir=None):
    path = f"{data_dir}/kfold_roc_curves_data.pkl"
    f = _load_pickle(path)
    if f is None:
        return

    if len(f['class_names']) > 2:
        f['class_names'][2] = 'Unlensed'

    plot_kfold_roc_curves(
        fold_results=f['fold_results'],
        class_names=f['class_names'],
        results_dir=results_dir if results_dir else './'
    )


def _plot_average_training_curves(data_dir, results_dir=None):
    path = f"{data_dir}/average_training_curves_kfold_data.pkl"
    f = _load_pickle(path)
    if f is None:
        return

    plot_average_training_curves(f['fold_results'], results_dir=results_dir if results_dir else './')

os.makedirs('./Plots', exist_ok=True)

data_configs = [
    ("ViT_results/plot_data", "ViT"),
    ("SimpleCNN_results/plot_data", "SimpleCNN"),
    ("Binary_SimpleCNN_results/plot_data", "Binary_SimpleCNN"),
    ("Binary_ViT_results/plot_data", "Binary_ViT"),
]


for data_dir, model_name in data_configs:
    print(f"Plotting results for {model_name} ({data_dir})...")

    _plot_confusion_matrix(data_dir, model_name, results_dir=data_dir)
    _plot_roc_curves(data_dir, model_name, results_dir=data_dir)
    _plot_training_curves(data_dir, model_name, results_dir=data_dir)

data_configs = [
    ("ViT_results/plot_data", "ViT"),
    ("SimpleCNN_results/plot_data", "SimpleCNN")
]

for data_dir, model_name in data_configs:
    print(f"Plotting results for {model_name} ({data_dir})...")
    
    _plot_kfold_roc_curves(data_dir, results_dir=data_dir)
    _plot_average_training_curves(data_dir, results_dir=data_dir)