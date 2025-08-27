import os
import glob
import argparse
from pathlib import Path

import torch

import random
import numpy as np

from gwtorch.modules.general_utils import (
    plot_training_curves,
)

from gwtorch.modules.ml_utils import (
    seed_everything,
    setup_directories,
    run_kfold_training,
    get_model,
    load_all_data,
    train_model,
    evaluate_and_plot
)

# ----------------- Constants -----------------
RANDOM_SEED = 42
BATCH_SIZE = 512
EPOCHS = 20
LR = 3e-5
GAMMA = 0.7
K_FOLDS = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN model for GW classification")
    parser.add_argument('--model', type=str, default='CNN', choices=['CNN'], help='Model to train')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--gamma', type=float, default=GAMMA)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--model-path', type=str, default='cnn_model.pth')
    parser.add_argument('--k-folds', type=int, default=K_FOLDS, help="Number of folds for k-fold cross validation")
    parser.add_argument('--use-kfold', action='store_true', help="Use k-fold cross validation instead of single train/val/test split")
    parser.add_argument('--path-name', type=str, default='data', help='Path to the dataset directory. Default is "data".')
    return parser.parse_args()


# ----------------- Main -----------------
def main():
    args = parse_args()
    setup_directories(model_name=args.model)
    seed_everything(args.seed)
    print(f"Using path {args.path_name}")
    print(f"Using device: {device}")

    if args.use_kfold:
        fold_results = run_kfold_training(args, device, get_model=get_model, model_name=args.model)
        print(f"\nK-fold cross validation completed!")
    else:
        print("\nRunning single train/validation/test split...")
        num_channels, train_loader, valid_loader, test_loader = load_all_data(path_name=args.path_name, seed=args.seed, batch_size=args.batch_size)

        model = get_model(name=args.model, num_channels=num_channels, device=device)

        print(f"\nNumber of channels in data: {num_channels}")
        
        if os.path.exists(f'./models/{args.model_path}'):
            print("Pretrained model found. Loading...\n")
            model.load_state_dict(torch.load(f'./models/{args.model_path}', map_location=device))
        else:
            print("No pretrained model found. Training from scratch...\n")

        train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, valid_loader, device, args.epochs, args.lr, args.gamma
        )
        val_misclassified, test_misclassified, _, _ = evaluate_and_plot(
            model, valid_loader, test_loader, device , model_name=args.model
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
