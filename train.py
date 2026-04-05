#!/usr/bin/env python
"""Training script for wildfire detection models.

Usage:
    python train.py --config configs/classical_baseline.yaml
    python train.py --config configs/classical_baseline.yaml --epochs 1 --batch_size 2
"""
import argparse
import os
import time
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils import data
from tqdm import tqdm
import yaml

from models import get_model, MODE_NAMES
from losses import get_loss
from dataset.sen2fire import Sen2FireDataSet
from utils.metrics import label_accuracy_score, eval_image
from utils.visualization import plot_training_history


def load_config(config_path, cli_overrides):
    """Load YAML config and apply CLI overrides."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value
    return config


def setup_experiment_dir(config):
    """Create timestamped experiment output directory."""
    timestamp = time.strftime('%m%d_%H%M', time.localtime())
    name = config.get('experiment_name', 'experiment')
    exp_dir = os.path.join('experiments', f"{name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    # Save a copy of the config for reproducibility
    shutil.copy2(config['_config_path'], os.path.join(exp_dir, 'config.yaml'))
    return exp_dir


def get_data_loaders(config):
    """Create train/val/test data loaders."""
    mode = config['mode']
    common = {'pin_memory': True, 'shuffle': True}

    # Auto-detect number of workers
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        n_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        n_workers = min(4, os.cpu_count() or 1)

    common['num_workers'] = n_workers

    train_loader = data.DataLoader(
        Sen2FireDataSet(config['data_dir'], config['train_list'], mode=mode,
                        augment=config.get('augment', False)),
        batch_size=config.get('batch_size', 16), **common)

    val_loader = data.DataLoader(
        Sen2FireDataSet(config['data_dir'], config['val_list'], mode=mode),
        batch_size=config.get('val_batch_size', 1),
        num_workers=n_workers, pin_memory=True, shuffle=False)

    test_loader = data.DataLoader(
        Sen2FireDataSet(config['data_dir'], config['test_list'], mode=mode),
        batch_size=config.get('test_batch_size', 50),
        num_workers=n_workers, pin_memory=True, shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate(model, loader, device, num_classes, interp):
    """Run evaluation on a data loader, return metrics dict."""
    epsilon = 1e-14
    model.eval()
    TP_all = np.zeros((num_classes, 1))
    FP_all = np.zeros((num_classes, 1))
    TN_all = np.zeros((num_classes, 1))
    FN_all = np.zeros((num_classes, 1))
    n_valid = 0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        image, label, _, _ = batch
        label = label.squeeze().numpy()
        image = image.float().to(device)
        with torch.no_grad():
            pred = model(image)
        _, pred = torch.max(interp(nn.functional.softmax(pred, dim=1)).detach(), 1)
        pred = pred.squeeze().data.cpu().numpy()
        TP, FP, TN, FN, n = eval_image(pred.reshape(-1), label.reshape(-1), num_classes)
        TP_all += TP
        FP_all += FP
        TN_all += TN
        FN_all += FN
        n_valid += n

    OA = np.sum(TP_all) / n_valid
    metrics = {'OA': OA}
    for i in range(num_classes):
        P = TP_all[i] / (TP_all[i] + FP_all[i] + epsilon)
        R = TP_all[i] / (TP_all[i] + FN_all[i] + epsilon)
        F1 = 2.0 * P * R / (P + R + epsilon)
        IoU = TP_all[i] / (TP_all[i] + FP_all[i] + FN_all[i] + epsilon)
        metrics[f'class{i}_P'] = P.item()
        metrics[f'class{i}_R'] = R.item()
        metrics[f'class{i}_F1'] = F1.item()
        metrics[f'class{i}_IoU'] = IoU.item()

    metrics['mF1'] = np.mean([metrics[f'class{i}_F1'] for i in range(num_classes)])
    metrics['mIoU'] = np.mean([metrics[f'class{i}_IoU'] for i in range(num_classes)])
    return metrics


def train(config):
    """Main training function."""
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'SLURMD_NODENAME' in os.environ:
        print(f"Running on {os.environ['SLURMD_NODENAME']} ({device})")
    else:
        print(f"Running on {device}")

    # Reproducibility
    torch.manual_seed(config.get('seed', 1234))

    # Setup
    exp_dir = setup_experiment_dir(config)
    log_file = open(os.path.join(exp_dir, 'Training_log.txt'), 'w')
    num_classes = config.get('n_classes', 2)
    name_classes = np.array(['non-fire', 'fire'], dtype=str)

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"Config: {config.get('experiment_name')}")
    log(f"Model: {config['model']} | Mode: {config['mode']} ({MODE_NAMES[config['mode']]}) | Augment: {config.get('augment', False)}")
    log(f"Output: {exp_dir}")

    # Model, optimizer, loss
    model = get_model(config)
    model = model.to(device)
    optimizer = Adam(model.parameters(),
                     lr=config.get('learning_rate', 1e-4),
                     weight_decay=config.get('weight_decay', 5e-4))
    loss_fn = get_loss(config)
    loss_fn = loss_fn.to(device)

    input_size = (512, 512)
    interp = nn.Upsample(size=input_size, mode='bilinear')

    # Data
    train_loader, val_loader, test_loader = get_data_loaders(config)
    log(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} | Test: {len(test_loader)}")

    # Training loop
    epochs = config.get('epochs', 5)
    hist = []
    F1_best = 0.0

    model.train()
    for epoch in range(1, epochs + 1):
        log(f"\n--- Epoch {epoch}/{epochs} ---")

        for batch_idx, (patches, labels, _, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            torch.cuda.empty_cache()
            start_time = time.time()

            patches = patches.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad()

            preds = interp(model(patches))
            loss = loss_fn(preds, labels)

            # Batch metrics
            _, pred_labels = torch.max(preds, 1)
            lbl_pred = pred_labels.detach().cpu().numpy()
            lbl_true = labels.detach().cpu().numpy()
            metrics_batch = []
            for lt, lp in zip(lbl_true, lbl_pred):
                _, _, mean_iu, _ = label_accuracy_score(lt, lp, n_class=num_classes)
                metrics_batch.append(mean_iu)

            batch_miou = np.nanmean(metrics_batch)
            batch_oa = np.sum(lbl_pred == lbl_true) / len(lbl_true.reshape(-1))
            elapsed = time.time() - start_time
            hist.append([loss.item(), batch_oa, batch_miou, elapsed])

            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                log(f"  Iter {batch_idx+1}/{len(train_loader)} | Loss: {hist[-1][0]:.4f} | OA: {hist[-1][1]:.4f} | mIoU: {hist[-1][2]:.4f} | Time: {hist[-1][3]:.2f}s")

        # Validation
        log("Validating...")
        val_metrics = evaluate(model, val_loader, device, num_classes, interp)
        log(f"  Val OA: {val_metrics['OA']*100:.2f}% | Fire F1: {val_metrics['class1_F1']*100:.2f}% | Fire IoU: {val_metrics['class1_IoU']*100:.2f}% | mIoU: {val_metrics['mIoU']*100:.2f}%")

        if val_metrics['class1_F1'] > F1_best:
            F1_best = val_metrics['class1_F1']
            log("  New best model! Saving checkpoint...")
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))

    # Test
    log("\nTesting...")
    best_path = os.path.join(exp_dir, 'best_model.pth')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, device, num_classes, interp)
    log(f"  Test OA: {test_metrics['OA']*100:.2f}% | Fire P: {test_metrics['class1_P']*100:.2f}% | Fire R: {test_metrics['class1_R']*100:.2f}% | Fire F1: {test_metrics['class1_F1']*100:.2f}% | Fire IoU: {test_metrics['class1_IoU']*100:.2f}% | mIoU: {test_metrics['mIoU']*100:.2f}%")

    log_file.close()

    # Save history
    np.savez(os.path.join(exp_dir, 'training_history.npz'), hist=hist)
    plot_training_history(hist, os.path.join(exp_dir, 'training_plot.png'))

    log_file = open(os.path.join(exp_dir, 'Training_log.txt'), 'a')
    log(f"\nDone. Results saved to {exp_dir}")
    log_file.close()


def main():
    parser = argparse.ArgumentParser(description='Train wildfire detection model')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    # CLI overrides (these take precedence over the config file)
    parser.add_argument('--mode', type=int, help='Spectral band mode (0-11)')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--fire_class_weight', type=int)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != 'config' and v is not None}
    config = load_config(args.config, overrides)
    config['_config_path'] = args.config
    train(config)


if __name__ == '__main__':
    main()
