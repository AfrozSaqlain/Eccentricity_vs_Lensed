
# ==============================================================================
# ROC CURVE AND AUC COMPUTATION WITH MISCLASSIFICATION TRACKING
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns

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

def plot_roc_curves(fpr, tpr, roc_auc, class_names, title_suffix="", results_dir=None):
    """
    Plot ROC curves for multi-class classification
    """

    title = title_suffix.split(" Set")[0].split("(")[1]

    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Plot ROC curve for each class
    for i, class_name in enumerate(class_names):
        if i in fpr:
            plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                    label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
    
    # Plot macro-average ROC curve
    plt.plot(fpr["macro"], tpr["macro"], color='black', linestyle='--', lw=2,
            label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - Multi-class Classification{title_suffix}', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / f'ROC_Curves_{title}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # return plt.gcf()

def plot_confusion_matrix(y_true, y_pred, class_names, title_suffix="", results_dir=None):
    """
    Plot confusion matrix with counts and percentages
    """

    title = title_suffix.split(" Set")[0].split("(")[1]

    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create annotations with both count and percentage
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = cm_percentage[i, j]
            row.append(f"{count}\n({percentage:.1f}%)")
        annotations.append(row)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix{title_suffix}', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(results_dir / f'Confusion_Matrix_{title}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # return plt.gcf()


def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, results_dir):
    print("Generating training curves...")
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs_range, val_losses, 'r-', label='Val Loss')
    plt.fill_between(epochs_range, train_losses, alpha=0.3, color='blue')
    plt.fill_between(epochs_range, val_losses, alpha=0.3, color='red')
    plt.title('Loss Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'Training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()