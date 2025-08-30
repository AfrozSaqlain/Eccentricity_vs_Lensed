
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

import matplotlib.pyplot as plt
from sklearn.metrics import auc

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

    all_keys = set().union(*(row.keys() for row in rows))
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_keys))
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
    all_img_paths = []
    
    with torch.no_grad():
        for data, labels, _, img_paths in data_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
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

def get_file_prefix(results_dir):
    """
    Get the file prefix for saving results based on the results directory.
    """
    file = str(results_dir)
    prefix = file.split('/')
    prefix1 = file.split('/')[1].split('_')
    prefix2 = None

    if 'fold' in prefix[-1]:
        prefix2 = file.split('/')[2].split('_')

    if len(prefix1) == 2:
    
        prefix1 = prefix1[0]
        if prefix2 is not None:
            prefix2 = prefix2[1]

        if prefix1 == 'ViT' and prefix2 is None:
            return 'ViT'
        elif prefix1 == 'CNN' and prefix2 is None:
            return 'CNN'
        elif prefix1 == 'ViT' and prefix2 is not None:
            return f'ViT_Fold_{prefix2}'
        elif prefix1 == 'CNN' and prefix2 is not None:
            return f'CNN_Fold_{prefix2}'

    elif len(prefix1) == 3:
        
        prefix1 = f"{prefix1[0]}_{prefix1[1]}"
        if prefix2 is not None:
            prefix2 = prefix2[1] 

        if prefix1 == 'Binary_ViT' and prefix2 is None:
            return 'Binary_ViT'
        elif prefix1 == 'Binary_CNN' and prefix2 is None:
            return 'Binary_CNN'
        elif prefix1 == 'Binary_ViT' and prefix2 is not None:
            return f'Binary_ViT_Fold_{prefix2}'
        elif prefix1 == 'Binary_CNN' and prefix2 is not None:
            return f'Binary_CNN_Fold_{prefix2}'

def plot_roc_curves(fpr, tpr, roc_auc, class_names, title_suffix="", results_dir=None, scale='linear'):

    prefix = get_file_prefix(results_dir) if results_dir else ''

    title = title_suffix.split(" Set")[0].split("(")[1]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Set scale-specific parameters
    if scale == 'log':
        eps = 1e-6
        xlabel = r'False Positive Rate (log scale)'
        xlim = [eps, 1.0]
        ylim = [eps, 1.05]
        diagonal = [eps, 1]
    else:  # linear scale
        eps = None
        xlabel = r'False Positive Rate'
        xlim = [0.0, 1.0]
        ylim = [0.0, 1.05]
        diagonal = [0, 1]

    for i, class_name in enumerate(class_names):
        if i in fpr:
            fpr_i = np.maximum(fpr[i], eps) if eps else fpr[i]
            tpr_i = tpr[i]
            ax.plot(fpr_i, tpr_i, color=colors[i % len(colors)], lw=2,
                    label=f'{class_name} (AUC = {roc_auc[i]:.3f})')

    fpr_macro = np.maximum(fpr["macro"], eps) if eps else fpr["macro"]
    ax.plot(fpr_macro, tpr["macro"], color='black', linestyle='--', lw=2,
            label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')

    ax.plot(diagonal, [0, 1], 'k--', lw=1, alpha=0.5)

    # Set scaling
    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
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


def plot_confusion_matrix(y_true, y_pred, class_names, title_suffix="", results_dir=None):

    prefix = get_file_prefix(results_dir) if results_dir else ''
    title = title_suffix.split(" Set")[0].split("(")[1]

    cm = confusion_matrix(y_true, y_pred)
    # cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_percentage = np.divide(
        cm.astype('float') * 100, 
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0
    )


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
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

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

    # epochs_range = range(1, len(train_losses) + 1)
    # axs[2].plot(epochs_range, train_losses, 'b-', label='Train Loss')
    # axs[2].plot(epochs_range, val_losses, 'r-', label='Val Loss')
    # axs[2].fill_between(epochs_range, train_losses, alpha=0.3, color='blue')
    # axs[2].fill_between(epochs_range, val_losses, alpha=0.3, color='red')
    # axs[2].set_title('Loss Progression')
    # axs[2].set_xlabel('Epoch')
    # axs[2].set_ylabel('Loss')
    # axs[2].legend()
    # axs[2].grid(True, alpha=0.3)
    # set_tick_sizes(axs[2])

    plt.tight_layout()
    results_dir = Path(results_dir)
    os.makedirs(results_dir / 'Plots', exist_ok=True)
    fig.savefig(results_dir / f'Plots/{prefix}_Training_curves.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plot_kfold_roc_curves(fold_results, results_dir, class_names):
    """Plot ROC curves for all folds with mean and std"""
        
    # Create figure with subplots for each class
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(fold_results['roc_data'])))
    
    for class_idx, class_name in enumerate(class_names):
        ax = axes[class_idx]
        
        # Common FPR values for interpolation
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        
        # Plot individual fold ROC curves
        for fold_idx, (fpr_dict, tpr_dict, auc_dict) in enumerate(fold_results['roc_data']):
            fpr = fpr_dict[class_idx]
            tpr = tpr_dict[class_idx]
            roc_auc = auc_dict[class_idx]
            
            # Interpolate TPR at common FPR values
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)
            
            # Plot individual fold
            ax.plot(fpr, tpr, alpha=0.3, color=colors[fold_idx], 
                    label=f'Fold {fold_idx+1} (AUC = {roc_auc:.3f})')
        
        # Calculate mean and std
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        # Plot mean ROC curve
        ax.plot(mean_fpr, mean_tpr, color='b', linewidth=1.5,
                label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
        
        # Plot confidence interval
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3,
                        label='± 1 std. dev.')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        # Customize plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves - {class_name}\nK-Fold Cross Validation')
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Store mean AUC for summary
        fold_results[f'mean_auc_{class_name.lower().replace(" ", "_")}'] = mean_auc
        fold_results[f'std_auc_{class_name.lower().replace(" ", "_")}'] = std_auc
    
    # Overall summary plot
    ax = axes[3]
    ax.axis('off')
    
    # Create summary text
    summary_text = "K-Fold ROC AUC Summary\n" + "="*25 + "\n\n"
    for class_idx, class_name in enumerate(class_names):
        class_key = class_name.lower().replace(" ", "_")
        mean_auc = fold_results[f'mean_auc_{class_key}']
        std_auc = fold_results[f'std_auc_{class_key}']
        summary_text += f"{class_name}:\n"
        summary_text += f"  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}\n"
        summary_text += f"  Individual AUCs: {[f'{auc:.3f}' for _, _, auc_dict in fold_results['roc_data'] for auc in [auc_dict[class_idx]]]}\n\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(results_dir / 'kfold_roc_curves_summary.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a simplified version with just mean curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    class_colors = ['red', 'green', 'blue']
    for class_idx, class_name in enumerate(class_names):
        # Recalculate mean for simplified plot
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        
        for fpr_dict, tpr_dict, auc_dict in fold_results['roc_data']:
            fpr = fpr_dict[class_idx]
            tpr = tpr_dict[class_idx]
            roc_auc = auc_dict[class_idx]
            
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        ax.plot(mean_fpr, mean_tpr, color=class_colors[class_idx], linewidth=1,
                label=f'{class_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Mean ROC Curves - K-Fold Cross Validation')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'kfold_mean_roc_curves.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("ROC curve plots saved successfully!")
    
def plot_average_training_curves(fold_results, results_dir):
    """Plot average training curves across all folds"""

    # Calculate means and stds
    train_losses_mean = np.mean(fold_results['train_losses_all'], axis=0)
    train_losses_std = np.std(fold_results['train_losses_all'], axis=0)
    val_losses_mean = np.mean(fold_results['val_losses_all'], axis=0)
    val_losses_std = np.std(fold_results['val_losses_all'], axis=0)
    
    train_accs_mean = np.mean(fold_results['train_accs_all'], axis=0)
    train_accs_std = np.std(fold_results['train_accs_all'], axis=0)
    val_accs_mean = np.mean(fold_results['val_accs_all'], axis=0)
    val_accs_std = np.std(fold_results['val_accs_all'], axis=0)
    
    epochs = range(1, len(train_losses_mean) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses_mean, 'b-', label='Training Loss')
    ax1.fill_between(epochs, train_losses_mean - train_losses_std, 
                    train_losses_mean + train_losses_std, alpha=0.2, color='blue')
    ax1.plot(epochs, val_losses_mean, 'r-', label='Validation Loss')
    ax1.fill_between(epochs, val_losses_mean - val_losses_std, 
                    val_losses_mean + val_losses_std, alpha=0.2, color='red')
    ax1.set_title('Average Training and Validation Loss (K-Fold)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs_mean, 'b-', label='Training Accuracy')
    ax2.fill_between(epochs, train_accs_mean - train_accs_std, 
                    train_accs_mean + train_accs_std, alpha=0.2, color='blue')
    ax2.plot(epochs, val_accs_mean, 'r-', label='Validation Accuracy')
    ax2.fill_between(epochs, val_accs_mean - val_accs_std, 
                    val_accs_mean + val_accs_std, alpha=0.2, color='red')
    ax2.set_title('Average Training and Validation Accuracy (K-Fold)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'average_training_curves_kfold.pdf', dpi=300, bbox_inches='tight')
    plt.close()
