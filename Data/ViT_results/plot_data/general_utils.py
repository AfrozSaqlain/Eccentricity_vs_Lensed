# ==============================================================================
# ROC CURVE AND AUC COMPUTATION WITH MISCLASSIFICATION TRACKING
# ==============================================================================

import os
import csv
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

import matplotlib as mpl

STYLE = {
    # --- Text & fonts ---
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'mathtext.default': 'sf',
    'font.size': 20,

    # --- Axes ---
    'axes.labelsize': 20,
    'axes.linewidth': 1.1,
    'axes.axisbelow': True,

    # --- Lines ---
    'lines.linewidth': 1.0,
    'lines.markeredgewidth': 1.0,
    'lines.markersize': 10.0,

    # --- Legend ---
    'legend.fontsize': 20,

    # --- Ticks: size & direction ---
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,

    'xtick.direction': 'in',
    'ytick.direction': 'in',

    'xtick.top': True,
    'ytick.right': True,

    'xtick.major.size': 12,
    'xtick.major.width': 1.5,
    'xtick.minor.size': 7,
    'xtick.minor.width': 1.0,

    'ytick.major.size': 12,
    'ytick.major.width': 1.5,
    'ytick.minor.size': 7,
    'ytick.minor.width': 1.0,

    'xtick.major.pad': 15,
    'ytick.major.pad': 15,

    # --- Figure ---
    'figure.figsize': (7.0, 4.5),
}

mpl.rcParams.update(STYLE)

def set_tick_sizes(ax, major=12, minor=7):
    for l in ax.get_xticklines() + ax.get_yticklines():
        l.set_markersize(major)
    for tick in ax.xaxis.get_minor_ticks() + ax.yaxis.get_minor_ticks():
        tick.tick1line.set_markersize(minor)
        tick.tick2line.set_markersize(minor)



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
        elif prefix1 == 'SimpleCNN' and prefix2 is None:
            return 'SimpleCNN'
        elif prefix1 == 'CvT' and prefix2 is None:
            return 'CvT'
        elif prefix1 == 'ViT' and prefix2 is not None:
            return f'ViT_Fold_{prefix2}'
        elif prefix1 == 'CNN' and prefix2 is not None:
            return f'CNN_Fold_{prefix2}'
        elif prefix1 == 'SimpleCNN' and prefix2 is not None:
            return f'SimpleCNN_Fold_{prefix2}'
        elif prefix1 == 'CvT' and prefix2 is not None:
            return f'CvT_Fold_{prefix2}'

    elif len(prefix1) == 3:
        
        prefix1 = f"{prefix1[0]}_{prefix1[1]}"
        if prefix2 is not None:
            prefix2 = prefix2[1] 

        if prefix1 == 'Binary_ViT' and prefix2 is None:
            return 'Binary_ViT'
        elif prefix1 == 'Binary_CNN' and prefix2 is None:
            return 'Binary_CNN'
        elif prefix1 == 'Binary_CvT' and prefix2 is None:
            return 'Binary_CvT'
        elif prefix1 == 'Binary_ViT' and prefix2 is not None:
            return f'Binary_ViT_Fold_{prefix2}'
        elif prefix1 == 'Binary_CNN' and prefix2 is not None:
            return f'Binary_CNN_Fold_{prefix2}'
        elif prefix1 == 'Binary_CvT' and prefix2 is not None:
            return f'Binary_CvT_Fold_{prefix2}'



def plot_roc_curves(fpr, tpr, roc_auc, class_names, title_suffix="", results_dir=None, scale='linear'):

    prefix = get_file_prefix(results_dir) if results_dir else ''
    title = title_suffix.split(" Set")[0].split("(")[1]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    if scale == 'log':
        eps = 1e-6
        xlabel = r'False Positive Rate (log scale)'
        xlim = [eps, 1.0]
        ylim = [eps, 1.05]
        diagonal = [eps, 1]
    else:
        eps = None
        xlabel = r'False Positive Rate'
        xlim = [0.0, 1.0]
        ylim = [0.0, 1.05]
        diagonal = [0, 1]

    for i, class_name in enumerate(class_names):
        if i in fpr:
            fpr_i = np.maximum(fpr[i], eps) if eps else fpr[i]
            tpr_i = tpr[i]
            ax.plot(fpr_i, tpr_i, color=colors[i % len(colors)],
                    label=rf'{class_name} (AUC = {roc_auc[i]:.3f})')

    fpr_macro = np.maximum(fpr["macro"], eps) if eps else fpr["macro"]
    ax.plot(fpr_macro, tpr["macro"], color='black', linestyle='--',
            label=rf'Macro-average (AUC = {roc_auc["macro"]:.3f})')

    ax.plot(diagonal, [0, 1], 'k--', alpha=0.5)

    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'True Positive Rate')
    ax.legend(loc="lower right")
    ax.grid(True, which='both', linestyle='--', alpha=0.3)

    set_tick_sizes(ax)

    plt.tight_layout()
    if results_dir:
        results_dir = Path(results_dir)
        os.makedirs(results_dir / 'Plots', exist_ok=True)
        fig.savefig(results_dir / f'Plots/{prefix}_ROC_Curves_{title}.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, title_suffix="", results_dir=None):

    prefix = get_file_prefix(results_dir) if results_dir else ''
    title = title_suffix.split(" Set")[0].split("(")[1]

    cm = confusion_matrix(y_true, y_pred)

    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_percentage = np.divide(
        cm.astype('float') * 100,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0
    )

    # ✅ LaTeX-safe annotations
    annotations = [[rf"${cm_percentage[i, j]:.1f}\%$"
                    for j in range(cm.shape[1])]
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
        annot_kws={"fontsize": 16},
        cbar_kws={"label": r"$\mathrm{Percentage\ (\%)}$"}
    )

    ax.set_xlabel(r'Predicted Label')
    ax.set_ylabel(r'True Label')

    set_tick_sizes(ax)

    ax.tick_params(axis='x', which='major', pad=15)
    ax.tick_params(axis='y', which='major', pad=15)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    plt.tight_layout()

    if results_dir:
        results_dir = Path(results_dir)
        os.makedirs(results_dir / 'Plots', exist_ok=True)
        fig.savefig(results_dir / f'Plots/{prefix}_Confusion_Matrix_{title}.pdf', dpi=300, bbox_inches='tight')

    plt.close()

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, results_dir=None):

    prefix = get_file_prefix(results_dir) if results_dir else ''

    fig, ax1 = plt.subplots(figsize=(8, 5))

    epochs = range(len(train_losses))

    l1 = ax1.plot(epochs, train_losses, label=r'Train Loss', linestyle='-')
    l2 = ax1.plot(epochs, val_losses, label=r'Validation Loss', linestyle='--')

    ax1.set_xlabel(r'Epoch')
    ax1.set_ylabel(r'Loss')
    ax1.grid(True, alpha=0.3)
    set_tick_sizes(ax1)

    ax2 = ax1.twinx()

    l3 = ax2.plot(epochs, train_accuracies, label=r'Train Accuracy', linestyle='-')
    l4 = ax2.plot(epochs, val_accuracies, label=r'Validation Accuracy', linestyle='--')

    ax2.set_ylabel(r'Accuracy')
    set_tick_sizes(ax2)

    lines = l1 + l2 + l3 + l4
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='center right')

    plt.tight_layout()

    if results_dir:
        results_dir = Path(results_dir)
        os.makedirs(results_dir / 'Plots', exist_ok=True)
        fig.savefig(results_dir / f'Plots/{prefix}_Training_curves.pdf', dpi=300, bbox_inches='tight')

    plt.close()
    
def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, results_dir=None):

    prefix = get_file_prefix(results_dir) if results_dir else ''

    fig, ax1 = plt.subplots(figsize=(8, 5))

    epochs = range(len(train_losses))

    l1 = ax1.plot(epochs, train_losses, label=r'Train Loss', linestyle='-')
    l2 = ax1.plot(epochs, val_losses, label=r'Validation Loss', linestyle='--')

    ax1.set_xlabel(r'Epoch')
    ax1.set_ylabel(r'Loss')
    ax1.grid(True, alpha=0.3)
    set_tick_sizes(ax1)

    ax2 = ax1.twinx()

    l3 = ax2.plot(epochs, train_accuracies, label=r'Train Accuracy', linestyle='-')
    l4 = ax2.plot(epochs, val_accuracies, label=r'Validation Accuracy', linestyle='--')

    ax2.set_ylabel(r'Accuracy')
    set_tick_sizes(ax2)

    lines = l1 + l2 + l3 + l4
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='center right')

    plt.tight_layout()

    if results_dir:
        results_dir = Path(results_dir)
        os.makedirs(results_dir / 'Plots', exist_ok=True)
        fig.savefig(results_dir / f'Plots/{prefix}_Training_curves.pdf', dpi=300, bbox_inches='tight')

    plt.close()


def plot_kfold_roc_curves(fold_results, results_dir=None, class_names=None):

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    colors = plt.cm.Set1(np.linspace(0, 1, len(fold_results['roc_data'])))

    for class_idx, class_name in enumerate(class_names):
        ax = axes[class_idx]

        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        for fold_idx, (fpr_dict, tpr_dict, auc_dict) in enumerate(fold_results['roc_data']):
            fpr = fpr_dict[class_idx]
            tpr = tpr_dict[class_idx]
            roc_auc_val = auc_dict[class_idx]

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc_val)

            ax.plot(fpr, tpr, alpha=0.3, color=colors[fold_idx],
                    label=rf'Fold {fold_idx+1} (AUC = {roc_auc_val:.3f})')

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        ax.plot(mean_fpr, mean_tpr, color='b', linewidth=1.5,
                label=rf'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})')

        std_tpr = np.std(tprs, axis=0)
        ax.fill_between(mean_fpr,
                        np.maximum(mean_tpr - std_tpr, 0),
                        np.minimum(mean_tpr + std_tpr, 1),
                        color='grey', alpha=0.3,
                        label=r'$\pm$ 1 std. dev.')

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel(r'False Positive Rate')
        ax.set_ylabel(r'True Positive Rate')
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if results_dir:
        plt.savefig(Path(results_dir) / 'kfold_roc_curves_summary.pdf', dpi=300, bbox_inches='tight')

    plt.close()


def plot_average_training_curves(fold_results, results_dir=None):

    train_losses_mean = np.mean(fold_results['train_losses_all'], axis=0)
    train_losses_std = np.std(fold_results['train_losses_all'], axis=0)
    val_losses_mean = np.mean(fold_results['val_losses_all'], axis=0)
    val_losses_std = np.std(fold_results['val_losses_all'], axis=0)

    train_accs_mean = np.mean(fold_results['train_accs_all'], axis=0)
    train_accs_std = np.std(fold_results['train_accs_all'], axis=0)
    val_accs_mean = np.mean(fold_results['val_accs_all'], axis=0)
    val_accs_std = np.std(fold_results['val_accs_all'], axis=0)

    epochs = range(1, len(train_losses_mean) + 1)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    l1 = ax1.plot(epochs, train_losses_mean, label=r'Training Loss')
    ax1.fill_between(epochs,
                     train_losses_mean - train_losses_std,
                     train_losses_mean + train_losses_std,
                     alpha=0.2)

    l2 = ax1.plot(epochs, val_losses_mean, linestyle='--', label=r'Validation Loss')
    ax1.fill_between(epochs,
                     val_losses_mean - val_losses_std,
                     val_losses_mean + val_losses_std,
                     alpha=0.2)

    ax1.set_xlabel(r'Epoch')
    ax1.set_ylabel(r'Loss')
    ax1.grid(True, alpha=0.3)
    set_tick_sizes(ax1)

    ax2 = ax1.twinx()

    l3 = ax2.plot(epochs, train_accs_mean, label=r'Training Accuracy')
    ax2.fill_between(epochs,
                     train_accs_mean - train_accs_std,
                     train_accs_mean + train_accs_std,
                     alpha=0.2)

    l4 = ax2.plot(epochs, val_accs_mean, linestyle='--', label=r'Validation Accuracy')
    ax2.fill_between(epochs,
                     val_accs_mean - val_accs_std,
                     val_accs_mean + val_accs_std,
                     alpha=0.2)

    ax2.set_ylabel(r'Accuracy')
    set_tick_sizes(ax2)

    lines = l1 + l2 + l3 + l4
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='center right')

    plt.tight_layout()

    if results_dir:
        plt.savefig(Path(results_dir) / 'average_training_curves_kfold.pdf',
                    dpi=300, bbox_inches='tight')

    plt.close()

