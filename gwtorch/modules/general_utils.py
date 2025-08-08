
# ==============================================================================
# ROC CURVE AND AUC COMPUTATION WITH MISCLASSIFICATION TRACKING
# ==============================================================================

import os
import csv
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import os
import csv

# At the top of the plotting script
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
rc('font', serif='times')
rc('mathtext', default='sf')
rc("lines", markeredgewidth=1)
rc("lines", linewidth=1)
rc('axes', labelsize=20)
rc("axes", linewidth=0.5)
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)
rc('legend', fontsize=10)
rc('ytick', right=True, direction='in')
rc('xtick', top=True, direction='in')
rc('xtick.major', pad=15)
rc('ytick.major', pad=15)
rc('xtick.major', size=12)
rc('ytick.major', size=12)
rc('xtick.minor', size=7)
rc('ytick.minor', size=7)

def set_tick_sizes(ax, major=12, minor=7):
    for l in ax.get_xticklines() + ax.get_yticklines():
        l.set_markersize(major)
    for tick in ax.xaxis.get_minor_ticks() + ax.yaxis.get_minor_ticks():
        tick.tick1line.set_markersize(minor)
        tick.tick2line.set_markersize(minor)


def save_csv(path, rows):
    """
    Save a list of dictionaries to a CSV file, appending if the file exists.
    If appending, adjust 'sample' field values to continue from the last maximum.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file to write to.

    rows : list of dict
        A list of dictionaries where each dictionary represents a row to write to the CSV file.
        Each dictionary must contain a 'sample' key with an integer value.

    Returns
    -------
    None

    Notes
    -----
    - If the CSV file exists, the function reads the last 'sample' value and offsets all new
      'sample' values to avoid duplication.
    - Assumes that the 'sample' key exists and is an integer.
    """
    if not rows:
        return

    file_exists = os.path.exists(path)
    sample_offset = 0

    if file_exists:
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            try:
                existing_samples = [int(row['sample']) for row in reader if 'sample' in row and row['sample'].isdigit()]
                if existing_samples:
                    sample_offset = max(existing_samples) + 1
            except KeyError:
                pass  # 'sample' column not found

    # Update sample index in new rows
    for i, row in enumerate(rows):
        row['sample'] = i + sample_offset

    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def compute_roc_auc_with_misclassifications(model, data_loader, device, class_names=['Lensed', 'Eccentric', 'Unlensed']):
    """
    Compute ROC curves and AUC scores for multi-class classification
    Also track misclassified files
    """
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_file_numbers = []
    all_img_paths = []
    
    with torch.no_grad():
        for data, labels, file_numbers, img_paths in data_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_file_numbers.extend(file_numbers)
            all_img_paths.extend(img_paths)
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Find misclassified samples
    misclassified_mask = all_labels != all_predictions
    misclassified_indices = np.where(misclassified_mask)[0]
    
    misclassified_info = []
    for idx in misclassified_indices:
        info = {
            'file_number': all_file_numbers[idx],
            'file_path': all_img_paths[idx],
            'true_label': all_labels[idx],
            'predicted_label': all_predictions[idx],
            'true_class': class_names[all_labels[idx]],
            'predicted_class': class_names[all_predictions[idx]],
            'prediction_probabilities': all_probabilities[idx]
        }
        misclassified_info.append(info)
    
    # Convert labels to binary format for ROC computation
    label_binarizer = LabelBinarizer()
    y_true_binary = label_binarizer.fit_transform(all_labels)
    
    # Handle case where we have only 2 classes (binary classification)
    if y_true_binary.shape[1] == 1:
        y_true_binary = np.hstack([1 - y_true_binary, y_true_binary])
    
    n_classes = y_true_binary.shape[1]
    
    # Compute ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], all_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute macro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    return fpr, tpr, roc_auc, all_labels, all_predictions, class_names, misclassified_info

def print_misclassified_files(misclassified_info, dataset_name):
    """
    Print information about misclassified files
    """
    print(f"\n{'='*60}")
    print(f"MISCLASSIFIED FILES - {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Total misclassified files: {len(misclassified_info)}")
    
    if len(misclassified_info) == 0:
        print("No misclassified files found!")
        return
    
    # Group by true class
    class_groups = {}
    for info in misclassified_info:
        true_class = info['true_class']
        if true_class not in class_groups:
            class_groups[true_class] = []
        class_groups[true_class].append(info)
    
    for true_class, files in class_groups.items():
        print(f"\n{'-'*40}")
        print(f"True Class: {true_class} ({len(files)} misclassified)")
        print(f"{'-'*40}")
        
        for info in files:
            print(f"File Number: {info['file_number']}")
            print(f"File Path: {info['file_path']}")
            print(f"Predicted as: {info['predicted_class']}")
            print(f"Prediction Probabilities: {info['prediction_probabilities']}")
            print(f"Max Probability: {info['prediction_probabilities'].max():.4f}")
            print()

def save_misclassified_files_to_txt(misclassified_info, dataset_name, filename):
    """
    Save misclassified files information to a text file
    """
    with open(filename, 'w') as f:
        f.write(f"MISCLASSIFIED FILES - {dataset_name.upper()}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total misclassified files: {len(misclassified_info)}\n\n")
        
        if len(misclassified_info) == 0:
            f.write("No misclassified files found!\n")
            return
        
        # Group by true class
        class_groups = {}
        for info in misclassified_info:
            true_class = info['true_class']
            if true_class not in class_groups:
                class_groups[true_class] = []
            class_groups[true_class].append(info)
        
        for true_class, files in class_groups.items():
            f.write(f"\n{'-'*40}\n")
            f.write(f"True Class: {true_class} ({len(files)} misclassified)\n")
            f.write(f"{'-'*40}\n")
            
            for info in files:
                f.write(f"File Number: {info['file_number']}\n")
                f.write(f"File Path: {info['file_path']}\n")
                f.write(f"Predicted as: {info['predicted_class']}\n")
                f.write(f"Prediction Probabilities: {info['prediction_probabilities']}\n")
                f.write(f"Max Probability: {info['prediction_probabilities'].max():.4f}\n")
                f.write("\n")


def save_misclassified_files_to_dict(misclassified_info, filename):
    """
    Save misclassified files information to a dictionary and save it to a file
    
    Args:
        misclassified_info: List of dictionaries containing misclassification info
        dataset_name: Name of the dataset
        filename: Path to save the dictionary file
    
    Returns:
        dict: Dictionary where keys are true classes and values are lists of filenames
    """
    # Initialize the result dictionary
    misclassified_dict = {}
    
    # If no misclassified files, return empty dict
    if len(misclassified_info) == 0:
        return misclassified_dict
    
    # Group by true class
    for info in misclassified_info:
        true_class = info['true_class']
        
        # Initialize list for this true class if not exists
        if true_class not in misclassified_dict:
            misclassified_dict[true_class] = []
        
        # Extract just the filename from the full path
        filename_only = info['file_path'].split('/')[-1]
        
        # Add just the filename to the appropriate true class
        misclassified_dict[true_class].append(filename_only)
    
    # Save the dictionary to a file
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(misclassified_dict, f)
    
    return misclassified_dict

# def plot_roc_curves(fpr, tpr, roc_auc, class_names, title_suffix="", results_dir=None):
#     """
#     Plot ROC curves for multi-class classification
#     """

#     title = title_suffix.split(" Set")[0].split("(")[1]

#     plt.figure(figsize=(12, 8))
    
#     colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
#     # Plot ROC curve for each class
#     for i, class_name in enumerate(class_names):
#         if i in fpr:
#             plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
#                     label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
    
#     # Plot macro-average ROC curve
#     plt.plot(fpr["macro"], tpr["macro"], color='black', linestyle='--', lw=2,
#             label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')
    
#     # Plot diagonal line
#     plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate', fontsize=12)
#     plt.ylabel('True Positive Rate', fontsize=12)
#     plt.title(f'ROC Curves - Multi-class Classification{title_suffix}', fontsize=14)
#     plt.legend(loc="lower right", fontsize=10)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(results_dir / f'ROC_Curves_{title}.pdf', dpi=300, bbox_inches='tight')
#     plt.show()
    
#     return plt.gcf()

# def plot_roc_curves(fpr, tpr, roc_auc, class_names, title_suffix="", results_dir=None):
#     """
#     Plot ROC curves for multi-class classification with log-scaled x-axis.
#     """

#     title = title_suffix.split(" Set")[0].split("(")[1]

#     plt.figure(figsize=(12, 8))

#     colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

#     # Small epsilon to avoid log(0)
#     eps = 1e-6

#     # Plot ROC curve for each class
#     for i, class_name in enumerate(class_names):
#         if i in fpr:
#             fpr_i = np.maximum(fpr[i], eps)
#             plt.plot(fpr_i, tpr[i], color=colors[i % len(colors)], lw=2,
#                      label=f'{class_name} (AUC = {roc_auc[i]:.3f})')

#     # Plot macro-average ROC curve
#     fpr_macro = np.maximum(fpr["macro"], eps)
#     plt.plot(fpr_macro, tpr["macro"], color='black', linestyle='--', lw=2,
#              label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')

#     # Diagonal reference line (not very meaningful on log scale, but often kept)
#     plt.plot([eps, 1], [0, 1], 'k--', lw=1, alpha=0.5)

#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlim([eps, 1.0])
#     plt.ylim([eps, 1.05])
#     plt.xlabel('False Positive Rate (log scale)', fontsize=12)
#     plt.ylabel('True Positive Rate', fontsize=12)
#     plt.title(f'ROC Curves - Multi-class Classification{title_suffix}', fontsize=14)
#     plt.legend(loc="lower right", fontsize=10)
#     plt.grid(True, which='both', linestyle='--', alpha=0.3)
#     plt.tight_layout()

#     if results_dir:
#         results_dir = Path(results_dir)
#         os.makedirs(results_dir / 'Plots', exist_ok=True)
#         plt.savefig(results_dir / f'Plots/ROC_Curves_{title}.pdf', dpi=300, bbox_inches='tight')
#     plt.show()


# def plot_confusion_matrix(y_true, y_pred, class_names, title_suffix="", results_dir=None):
#     """
#     Plot confusion matrix with counts and percentages
#     """

#     title = title_suffix.split(" Set")[0].split("(")[1]

#     cm = confusion_matrix(y_true, y_pred)
#     cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

#     annotations = []
#     for i in range(cm.shape[0]):
#         row = []
#         for j in range(cm.shape[1]):
#             count = cm[i, j]
#             percentage = cm_percentage[i, j]
#             row.append(f"{count}\n({percentage:.1f}%)")
#         annotations.append(row)

#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
#                xticklabels=class_names, yticklabels=class_names)
#     plt.title(f'Confusion Matrix{title_suffix}', fontsize=14)
#     plt.xlabel('Predicted Label', fontsize=12)
#     plt.ylabel('True Label', fontsize=12)
#     plt.tight_layout()
#     if results_dir is not None:
#         plt.savefig(results_dir / f'Plots/Confusion_Matrix_{title}.pdf', dpi=300, bbox_inches='tight')
#     plt.show()

#     # return plt.gcf()


# def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, results_dir):
#     print("Generating training curves...")
#     plt.figure(figsize=(15, 5))

#     plt.subplot(1, 3, 1)
#     plt.plot(train_losses, label='Train Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     plt.subplot(1, 3, 2)
#     plt.plot(train_accuracies, label='Train Accuracy')
#     plt.plot(val_accuracies, label='Validation Accuracy')
#     plt.title('Training and Validation Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     plt.subplot(1, 3, 3)
#     epochs_range = range(1, len(train_losses) + 1)
#     plt.plot(epochs_range, train_losses, 'b-', label='Train Loss')
#     plt.plot(epochs_range, val_losses, 'r-', label='Val Loss')
#     plt.fill_between(epochs_range, train_losses, alpha=0.3, color='blue')
#     plt.fill_between(epochs_range, val_losses, alpha=0.3, color='red')
#     plt.title('Loss Progression')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig(results_dir / 'Plots/Training_curves.pdf', dpi=300, bbox_inches='tight')
#     plt.show()

def get_file_prefix(results_dir):
    """
    Get the file prefix for saving results based on the results directory.
    """
    file = str(results_dir)
    prefix = file.split('/')[-1].split('_')[0]
    if prefix == 'transformer':
        return 'ViT'
    elif prefix == 'cnn':
        return 'CNN'

def plot_roc_curves(fpr, tpr, roc_auc, class_names, title_suffix="", results_dir=None):

    prefix = get_file_prefix(results_dir) if results_dir else ''

    title = title_suffix.split(" Set")[0].split("(")[1]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    eps = 1e-6

    for i, class_name in enumerate(class_names):
        if i in fpr:
            fpr_i = np.maximum(fpr[i], eps)
            ax.plot(fpr_i, tpr[i], color=colors[i % len(colors)], lw=2,
                    label=f'{class_name} (AUC = {roc_auc[i]:.3f})')

    fpr_macro = np.maximum(fpr["macro"], eps)
    ax.plot(fpr_macro, tpr["macro"], color='black', linestyle='--', lw=2,
            label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')

    ax.plot([eps, 1], [0, 1], 'k--', lw=1, alpha=0.5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([eps, 1.0])
    ax.set_ylim([eps, 1.05])
    ax.set_xlabel(r'False Positive Rate (log scale)')
    ax.set_ylabel(r'True Positive Rate')
    ax.set_title(f'ROC Curves - Multi-class Classification{title_suffix}')
    ax.legend(loc="lower right")
    ax.grid(True, which='both', linestyle='--', alpha=0.3)

    set_tick_sizes(ax)

    plt.tight_layout()
    if results_dir:
        results_dir = Path(results_dir)
        os.makedirs(results_dir / 'Plots', exist_ok=True)
        fig.savefig(results_dir / f'Plots/{prefix}_ROC_Curves_{title}.pdf', dpi=300, bbox_inches='tight')
    plt.show()


# def plot_confusion_matrix(y_true, y_pred, class_names, title_suffix="", results_dir=None):

#     title = title_suffix.split(" Set")[0].split("(")[1]

#     cm = confusion_matrix(y_true, y_pred)
#     cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

#     annotations = [[f"{cm[i, j]}\n({cm_percentage[i, j]:.1f}\\%)" for j in range(cm.shape[1])] for i in range(cm.shape[0])]

#     fig, ax = plt.subplots(figsize=(10, 8))
#     sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
#                 xticklabels=class_names, yticklabels=class_names, ax=ax)
#     ax.set_title(f'Confusion Matrix{title_suffix}')
#     ax.set_xlabel('Predicted Label')
#     ax.set_ylabel('True Label')
#     set_tick_sizes(ax)

#     plt.tight_layout()
#     if results_dir:
#         results_dir = Path(results_dir)
#         os.makedirs(results_dir / 'Plots', exist_ok=True)
#         fig.savefig(results_dir / f'Plots/Confusion_Matrix_{title}.pdf', dpi=300, bbox_inches='tight')
#     plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, title_suffix="", results_dir=None):

    prefix = get_file_prefix(results_dir) if results_dir else ''
    title = title_suffix.split(" Set")[0].split("(")[1]

    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Only percentages as annotations (escaped % for LaTeX)
    annotations = [[f"{cm_percentage[i, j]:.1f}\\%" for j in range(cm.shape[1])]
                   for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_percentage,
        annot=annotations,
        fmt='',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={"fontsize": 12},
        cbar_kws={"label": "Percentage (\\%)"}
    )

    ax.set_title(f'Confusion Matrix{title_suffix}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    set_tick_sizes(ax)
    set_tick_sizes(ax, major=12, minor=7)
    ax.tick_params(axis='x', which='major', pad=15)
    ax.tick_params(axis='y', which='major', pad=15)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    plt.tight_layout()
    if results_dir:
        results_dir = Path(results_dir)
        os.makedirs(results_dir / 'Plots', exist_ok=True)
        fig.savefig(results_dir / f'Plots/{prefix}_Confusion_Matrix_{title}.pdf', dpi=300, bbox_inches='tight')
    plt.show()




def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, results_dir):

    prefix = get_file_prefix(results_dir) if results_dir else ''

    print("Generating training curves...")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].plot(train_losses, label='Train Loss')
    axs[0].plot(val_losses, label='Validation Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    set_tick_sizes(axs[0])

    axs[1].plot(train_accuracies, label='Train Accuracy')
    axs[1].plot(val_accuracies, label='Validation Accuracy')
    axs[1].set_title('Training and Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    set_tick_sizes(axs[1])

    epochs_range = range(1, len(train_losses) + 1)
    axs[2].plot(epochs_range, train_losses, 'b-', label='Train Loss')
    axs[2].plot(epochs_range, val_losses, 'r-', label='Val Loss')
    axs[2].fill_between(epochs_range, train_losses, alpha=0.3, color='blue')
    axs[2].fill_between(epochs_range, val_losses, alpha=0.3, color='red')
    axs[2].set_title('Loss Progression')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Loss')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    set_tick_sizes(axs[2])

    plt.tight_layout()
    results_dir = Path(results_dir)
    os.makedirs(results_dir / 'Plots', exist_ok=True)
    fig.savefig(results_dir / f'Plots/{prefix}_Training_curves.pdf', dpi=300, bbox_inches='tight')
    plt.show()