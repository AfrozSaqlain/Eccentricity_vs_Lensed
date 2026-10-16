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

def set_tick_sizes(ax, major=12, minor=7):
    for l in ax.get_xticklines() + ax.get_yticklines():
        l.set_markersize(major)
    for tick in ax.xaxis.get_minor_ticks() + ax.yaxis.get_minor_ticks():
        tick.tick1line.set_markersize(minor)
        tick.tick2line.set_markersize(minor)



def get_file_prefix(results_dir):
    """
    Extract model name directly from results_dir.
    Expected formats:
        ViT_results/plot_data
        SimpleCNN_results/plot_data
        Binary_ViT_results/plot_data
        .../fold_X/... (optional)
    """

    parts = str(results_dir).split('/')

    # Find the part containing '_results'
    model_part = None
    for p in parts:
        if '_results' in p:
            model_part = p
            break

    if model_part is None:
        raise ValueError(f"Cannot extract model name from: {results_dir}")

    # Extract model name
    model_name = model_part.replace('_results', '')

    # Handle fold case if present
    for p in parts:
        if 'fold' in p:
            fold_idx = p.split('_')[-1]
            return f"{model_name}_Fold_{fold_idx}"

    return model_name



def plot_roc_curves(fpr, tpr, roc_auc, class_names, title_suffix="", results_dir=None, scale='linear'):

    STYLE = {
        # --- Text & fonts ---
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern'],
        'mathtext.default': 'sf',
        'font.size': 20,

        # --- Axes ---
        'axes.labelsize': 30,
        'axes.linewidth': 1.5,
        'axes.axisbelow': True,

        # --- Lines ---
        'lines.linewidth': 2.5,
        'lines.markeredgewidth': 1.0,
        'lines.markersize': 10.0,

        # --- Legend ---
        'legend.fontsize': 20,

        # --- Ticks: size & direction ---
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,

        'xtick.direction': 'in',
        'ytick.direction': 'in',

        'xtick.top': True,
        'ytick.right': True,

        'xtick.major.size': 14,
        'xtick.major.width': 1.5,
        'xtick.minor.size': 7,
        'xtick.minor.width': 1.0,

        'ytick.major.size': 14,
        'ytick.major.width': 1.5,
        'ytick.minor.size': 7,
        'ytick.minor.width': 1.0,

        'xtick.major.pad': 17,
        'ytick.major.pad': 17,

        # --- Figure ---
        'figure.figsize': (7.0, 4.5),
    }


    STYLE = {
                # --- Text & fonts ---
                'text.usetex': True,
                'font.family': 'serif',
                'font.serif': ['Computer Modern'],
                'mathtext.default': 'sf',
                'font.size': 20,
        
                # --- Axes ---
                'axes.labelsize': 30,
                'axes.linewidth': 1.5,
                'axes.axisbelow': True,
        
                # --- Lines ---
                'lines.linewidth': 2.7,
                'lines.markeredgewidth': 1.0,
                'lines.markersize': 10.0,
        
                # --- Legend ---
                'legend.fontsize': 28,
        
                # --- Ticks: size & direction ---
                'xtick.labelsize': 20,
                'ytick.labelsize': 20,
        
                'xtick.direction': 'in',
                'ytick.direction': 'in',
        
                'xtick.top': True,
                'ytick.right': True,
        
                'xtick.major.size': 14,
                'xtick.major.width': 1.5,
                'xtick.minor.size': 7,
                'xtick.minor.width': 1.0,
        
                'ytick.major.size': 14,
                'ytick.major.width': 1.5,
                'ytick.minor.size': 7,
                'ytick.minor.width': 1.0,
        
                'xtick.major.pad': 17,
                'ytick.major.pad': 17,
        
                # --- Figure ---
                'figure.figsize': (7.0, 4.5),
            }

    mpl.rcParams.update(STYLE)

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
                    label=rf'{class_name.capitalize()} (AUC = {roc_auc[i]:.3f})')

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

    save_dir = Path('./Plots') / prefix
    os.makedirs(save_dir, exist_ok=True)
    
    fig.savefig(save_dir / f'{prefix}_ROC_Curves_{title}.pdf',
                dpi=300, bbox_inches='tight')


def plot_confusion_matrix(y_true, y_pred, class_names, title_suffix="", results_dir=None):

    STYLE = {
        # --- Text & fonts ---
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern'],
        'mathtext.default': 'sf',
        'font.size': 20,

        # --- Axes ---
        'axes.labelsize': 22,
        'axes.linewidth': 1.5,
        'axes.axisbelow': True,

        # --- Lines ---
        'lines.linewidth': 1.0,
        'lines.markeredgewidth': 1.0,
        'lines.markersize': 10.0,

        # --- Legend ---
        'legend.fontsize': 20,

        # --- Ticks: size & direction ---
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,

        'xtick.direction': 'in',
        'ytick.direction': 'in',

        'xtick.top': True,
        'ytick.right': True,

        'xtick.major.size': 14,
        'xtick.major.width': 1.5,
        'xtick.minor.size': 7,
        'xtick.minor.width': 1.0,

        'ytick.major.size': 14,
        'ytick.major.width': 1.5,
        'ytick.minor.size': 7,
        'ytick.minor.width': 1.0,

        'xtick.major.pad': 17,
        'ytick.major.pad': 17,

        # --- Figure ---
        'figure.figsize': (7.0, 4.5),
    }

    mpl.rcParams.update(STYLE)

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

    save_dir = Path('./Plots') / prefix
    os.makedirs(save_dir, exist_ok=True)
    
    fig.savefig(save_dir / f'{prefix}_Confusion_Matrix_{title}.pdf',
                dpi=300, bbox_inches='tight')

    plt.close()

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, results_dir=None):

    STYLE = {
        # --- Text & fonts ---
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern'],
        'mathtext.default': 'sf',
        'font.size': 20,

        # --- Axes ---
        'axes.labelsize': 24,
        'axes.linewidth': 1.5,
        'axes.axisbelow': True,

        # --- Lines ---
        'lines.linewidth': 2.0,
        'lines.markeredgewidth': 1.0,
        'lines.markersize': 10.0,

        # --- Legend ---
        'legend.fontsize': 18,

        # --- Ticks: size & direction ---
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,

        'xtick.direction': 'in',
        'ytick.direction': 'in',

        'xtick.top': True,
        'ytick.right': True,

        'xtick.major.size': 14,
        'xtick.major.width': 1.5,
        'xtick.minor.size': 7,
        'xtick.minor.width': 1.0,

        'ytick.major.size': 14,
        'ytick.major.width': 1.5,
        'ytick.minor.size': 7,
        'ytick.minor.width': 1.0,

        'xtick.major.pad': 17,
        'ytick.major.pad': 17,

        # --- Figure ---
        'figure.figsize': (7.0, 4.5),
    }

    mpl.rcParams.update(STYLE)

    COLOR_MAP = {
        'train_loss': '#1f77b4',   # blue
        'val_loss': '#ff7f0e',     # orange
        'train_acc': '#2ca02c',    # green
        'val_acc': '#d62728'       # red
    }

    prefix = get_file_prefix(results_dir) if results_dir else ''

    fig, ax1 = plt.subplots(figsize=(8, 5))

    epochs = range(len(train_losses))

    l1 = ax1.plot(epochs, train_losses, label='Train Loss', linestyle='-', color=COLOR_MAP['train_loss'])
    l2 = ax1.plot(epochs, val_losses, label='Validation Loss', linestyle='--', color=COLOR_MAP['val_loss'])

    ax1.set_xlabel(r'Epoch')
    ax1.set_ylabel(r'Loss')
    ax1.grid(True, alpha=0.3)
    set_tick_sizes(ax1)

    ax2 = ax1.twinx()

    l3 = ax2.plot(epochs, train_accuracies, label='Train Accuracy', linestyle='-', color=COLOR_MAP['train_acc'])
    l4 = ax2.plot(epochs, val_accuracies, label='Validation Accuracy', linestyle='--', color=COLOR_MAP['val_acc'])

    ax2.set_ylabel(r'Accuracy')
    set_tick_sizes(ax2)

    lines = l1 + l2 + l3 + l4
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='center right')

    plt.tight_layout()

    save_dir = Path('./Plots') / prefix
    os.makedirs(save_dir, exist_ok=True)
    
    fig.savefig(save_dir / f'{prefix}_Training_curves.pdf',
                dpi=300, bbox_inches='tight')

    plt.close()

def plot_kfold_roc_curves(fold_results, results_dir=None, class_names=None):

    STYLE = {
        # --- Text & fonts ---
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern'],
        'mathtext.default': 'sf',
        'font.size': 20,

        # --- Axes ---
        'axes.labelsize': 22,
        'axes.linewidth': 1.5,
        'axes.axisbelow': True,

        # --- Lines ---
        'lines.linewidth': 2.5,
        'lines.markeredgewidth': 1.0,
        'lines.markersize': 10.0,

        # --- Legend ---
        'legend.fontsize': 20,

        # --- Ticks: size & direction ---
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,

        'xtick.direction': 'in',
        'ytick.direction': 'in',

        'xtick.top': True,
        'ytick.right': True,

        'xtick.major.size': 14,
        'xtick.major.width': 1.5,
        'xtick.minor.size': 7,
        'xtick.minor.width': 1.0,

        'ytick.major.size': 14,
        'ytick.major.width': 1.5,
        'ytick.minor.size': 7,
        'ytick.minor.width': 1.0,

        'xtick.major.pad': 17,
        'ytick.major.pad': 17,

        # --- Figure ---
        'figure.figsize': (7.0, 4.5),
    }

    mpl.rcParams.update(STYLE)

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
        
        ax.plot(mean_fpr, mean_tpr, color=class_colors[class_idx],
                label=rf'{class_name} (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    prefix = get_file_prefix(results_dir) if results_dir else 'Unknown'
    save_dir = Path('./Plots') / prefix
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_dir / f'{prefix}_kfold_mean_roc_curves.pdf',
                dpi=300, bbox_inches='tight')

    plt.close()


def plot_average_training_curves(fold_results, results_dir=None):

    STYLE = {
        # --- Text & fonts ---
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern'],
        'mathtext.default': 'sf',
        'font.size': 20,

        # --- Axes ---
        'axes.labelsize': 22,
        'axes.linewidth': 1.5,
        'axes.axisbelow': True,

        # --- Lines ---
        'lines.linewidth': 2.0,
        'lines.markeredgewidth': 1.0,
        'lines.markersize': 10.0,

        # --- Legend ---
        'legend.fontsize': 20,

        # --- Ticks: size & direction ---
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,

        'xtick.direction': 'in',
        'ytick.direction': 'in',

        'xtick.top': True,
        'ytick.right': True,

        'xtick.major.size': 14,
        'xtick.major.width': 1.5,
        'xtick.minor.size': 7,
        'xtick.minor.width': 1.0,

        'ytick.major.size': 14,
        'ytick.major.width': 1.5,
        'ytick.minor.size': 7,
        'ytick.minor.width': 1.0,

        'xtick.major.pad': 17,
        'ytick.major.pad': 17,

        # --- Figure ---
        'figure.figsize': (7.0, 4.5),
    }

    mpl.rcParams.update(STYLE)

    COLOR_MAP = {
        'train_loss': '#1f77b4',   # blue
        'val_loss': '#ff7f0e',     # orange
        'train_acc': '#2ca02c',    # green
        'val_acc': '#d62728'       # red
    }

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

    l1 = ax1.plot(epochs, train_losses_mean, label='Training Loss',
              linestyle='-', color=COLOR_MAP['train_loss'])
    ax1.fill_between(epochs,
                    train_losses_mean - train_losses_std,
                    train_losses_mean + train_losses_std,
                    alpha=0.2, color=COLOR_MAP['train_loss'])

    l2 = ax1.plot(epochs, val_losses_mean, label='Validation Loss',
                linestyle='--', color=COLOR_MAP['val_loss'])
    ax1.fill_between(epochs,
                    val_losses_mean - val_losses_std,
                    val_losses_mean + val_losses_std,
                    alpha=0.2, color=COLOR_MAP['val_loss'])

    ax1.set_xlabel(r'Epoch')
    ax1.set_ylabel(r'Loss')
    ax1.grid(True, alpha=0.3)
    set_tick_sizes(ax1)

    ax2 = ax1.twinx()

    l3 = ax2.plot(epochs, train_accs_mean, label='Training Accuracy',
              linestyle='-', color=COLOR_MAP['train_acc'])
    ax2.fill_between(epochs,
                    train_accs_mean - train_accs_std,
                    train_accs_mean + train_accs_std,
                    alpha=0.2, color=COLOR_MAP['train_acc'])

    l4 = ax2.plot(epochs, val_accs_mean, label='Validation Accuracy',
                linestyle='--', color=COLOR_MAP['val_acc'])
    ax2.fill_between(epochs,
                    val_accs_mean - val_accs_std,
                    val_accs_mean + val_accs_std,
                    alpha=0.2, color=COLOR_MAP['val_acc'])

    ax2.set_ylabel(r'Accuracy')
    set_tick_sizes(ax2)

    lines = l1 + l2 + l3 + l4
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='center right')

    plt.tight_layout()

    prefix = get_file_prefix(results_dir) if results_dir else 'Unknown'
    save_dir = Path('./Plots') / prefix
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_dir / f'{prefix}_average_training_curves_kfold.pdf',
                dpi=300, bbox_inches='tight')

    plt.close()

