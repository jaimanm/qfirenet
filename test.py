#!/usr/bin/env python
"""Inference script for wildfire detection models.

Usage:
    python test.py --config configs/classical_baseline.yaml \
        --restore_from experiments/classical_baseline_0815_1330/best_model.pth
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
from scipy import sparse
import yaml

from models import get_model, MODE_NAMES
from dataset.sen2fire import Sen2FireDataSet
from utils.metrics import eval_image
from utils.visualization import plot_scene_map

SCENE_DIMS = {1: (32, 27), 2: (22, 27), 3: (14, 36), 4: (21, 24)}
SCENE_LIST_FILES = {
    1: './dataset/train.txt',
    2: './dataset/train.txt',
    3: './dataset/val.txt',
    4: './dataset/test.txt',
}

epsilon = 1e-14
name_classes = np.array(['non-fire', 'fire'], dtype=str)


def load_config(config_path, cli_overrides):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value
    return config


def process_scene(scene_id, model, config, output_dir, device):
    """Generate predictions and build scene map for one scene."""
    scene_list = SCENE_LIST_FILES[scene_id]
    if not os.path.exists(scene_list):
        print(f"List file {scene_list} not found, skipping scene {scene_id}.")
        return

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', min(4, os.cpu_count() or 1)))
    input_size = (512, 512)
    interp = nn.Upsample(size=input_size, mode='bilinear').to(device)

    loader = data.DataLoader(
        Sen2FireDataSet(config['data_dir'], scene_list, mode=config['mode']),
        batch_size=1, shuffle=False, num_workers=n_workers, pin_memory=True)

    preds_dir = os.path.join(output_dir, 'preds', f'Scene{scene_id}')
    os.makedirs(preds_dir, exist_ok=True)

    # Generate predictions
    for batch in tqdm(loader, desc=f"Scene {scene_id} predictions"):
        image, _, _, name = batch
        image = image.float().to(device)
        patch_name = name[0].split('/')[1]
        filename = patch_name[:6] + str(scene_id) + patch_name[7:]
        patch_path = os.path.join(preds_dir, filename)
        if os.path.exists(patch_path + '.npz'):
            continue
        with torch.no_grad():
            pred = model(image)
        _, pred = torch.max(interp(nn.functional.softmax(pred, dim=1)).detach(), 1)
        pred = pred.squeeze().data.cpu().numpy().astype('uint8')
        sparse.save_npz(patch_path, sparse.csr_matrix(pred))

    # Reconstruct scene map
    n_row, n_col = SCENE_DIMS[scene_id]
    patch_size, overlap = 512, 128
    h = n_row * (patch_size - overlap) + overlap
    w = n_col * (patch_size - overlap) + overlap
    reconstructed_rgb = np.zeros((3, h, w))
    reconstructed_label = np.zeros((h, w))
    reconstructed_pred = np.zeros((h, w))

    image_dir = os.path.join(config['data_dir'], f"scene{scene_id}")
    for row in tqdm(range(1, n_row + 1), desc=f"Scene {scene_id} map"):
        for col in range(1, n_col + 1):
            patch_name = f"scene_{scene_id}_patch_{row}_{col}.npz"
            patch_data = np.load(os.path.join(image_dir, patch_name))['image']
            patch_gt = np.load(os.path.join(image_dir, patch_name))['label']
            filename = patch_name[:6] + str(scene_id) + patch_name[7:]
            patch_pred = sparse.load_npz(os.path.join(preds_dir, filename)).toarray()

            sr = (row - 1) * (patch_size - overlap)
            sc = (col - 1) * (patch_size - overlap)
            reconstructed_rgb[:, sr:sr+patch_size, sc:sc+patch_size] = patch_data[[3,2,1], :, :]
            reconstructed_label[sr:sr+patch_size, sc:sc+patch_size] = patch_gt

            # Handle overlap regions
            r_start = sr + (overlap // 2 if row > 1 else 0)
            c_start = sc + (overlap // 2 if col > 1 else 0)
            pr_start = overlap // 2 if row > 1 else 0
            pc_start = overlap // 2 if col > 1 else 0
            reconstructed_pred[r_start:sr+patch_size, c_start:sc+patch_size] = patch_pred[pr_start:, pc_start:]

    save_path = os.path.join(output_dir, f"scene{scene_id}_map.png")
    plot_scene_map(reconstructed_rgb, reconstructed_pred, reconstructed_label, save_path)


def main():
    parser = argparse.ArgumentParser(description='Run inference for wildfire detection')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    parser.add_argument('--restore_from', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--output_dir', type=str, help='Output directory (default: same as checkpoint dir)')
    parser.add_argument('--mode', type=int)
    parser.add_argument('--data_dir', type=str)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k not in ('config', 'restore_from', 'output_dir') and v is not None}
    config = load_config(args.config, overrides)

    # Output directory defaults to the experiment directory containing the checkpoint
    output_dir = args.output_dir or os.path.dirname(args.restore_from)
    os.makedirs(output_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = get_model(config)
    state_dict = torch.load(args.restore_from, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Model: {config['model']} | Mode: {config['mode']} ({MODE_NAMES[config['mode']]})")
    print(f"Checkpoint: {args.restore_from}")
    print(f"Output: {output_dir}")

    for scene_id in [1, 2, 3, 4]:
        process_scene(scene_id, model, config, output_dir, device)

    print(f"\nDone. Results saved to {output_dir}")


if __name__ == '__main__':
    main()
