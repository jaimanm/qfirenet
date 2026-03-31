# Wildfire Detection Using Quantum-Enhanced U-Net

Hybrid quantum-classical machine learning for wildfire detection using Sentinel-2 satellite imagery and the [Sen2Fire](https://arxiv.org/abs/2403.17884) dataset.

## Quick Start

```bash
pip install -r requirements.txt

# Train classical baseline
python train.py --config configs/classical_baseline.yaml

# Train quantum model
python train.py --config configs/quantum_strongly_entangling.yaml

# Run inference
python test.py --config configs/classical_baseline.yaml \
    --restore_from experiments/<run_dir>/best_model.pth
```

## Codebase Structure

```
configs/          YAML experiment configs
models/           Model architectures (classical U-Net, quantum U-Net)
circuits/         Quantum circuits (swappable via config)
losses/           Loss functions (swappable via config)
dataset/          Sen2Fire PyTorch dataset + split files
utils/            Metrics and visualization helpers
experiments/      Training run outputs (logs, plots, maps)
scripts/          SLURM job submission scripts
notebooks/        Shared reference notebooks
```

**Adding a new model, circuit, or loss function** = create a new file + one line in the registry. See [SOW.md](SOW.md) Section 5 for details.

## Dataset

Download the [Sen2Fire dataset](https://arxiv.org/abs/2403.17884) and place it at `../Sen2Fire/` relative to the repo root:

```
../Sen2Fire/
├── scene1/
│   ├── scene_1_patch_1_1.npz
│   └── ...
├── scene2/
├── scene3/
└── scene4/
```

Each `.npz` file contains `image` (12 bands, 512x512), `label` (binary mask), and `aerosol` data.

## Configuration

Experiments are fully described by YAML config files. Example:

```yaml
model: classical_unet    # or quantum_unet
circuit: strongly_entangling  # only for quantum_unet
mode: 5                   # SWIR + aerosol (best performing)
epochs: 5
batch_size: 16
learning_rate: 1.0e-4
loss: cross_entropy
fire_class_weight: 10
experiment_name: my_experiment
```

CLI arguments override config values: `python train.py --config configs/x.yaml --epochs 1`

## HPC (SLURM)

```bash
sbatch scripts/submit_train.sh --config configs/classical_baseline.yaml
sbatch scripts/submit_test.sh --config configs/classical_baseline.yaml --restore_from experiments/.../best_model.pth
```

## Baseline Results (Classical U-Net, Mode 5: SWIR + Aerosol)

| Metric | Value |
|--------|-------|
| Overall Accuracy | 93.41% |
| Fire Precision | 38.41% |
| Fire Recall | 23.18% |
| Fire F1-Score | 28.91% |
| Fire IoU | 16.90% |
| Mean IoU | 62.73% |

## Contributing

See [SOW.md](SOW.md) for the full workflow, coding guidelines, and how to add new components. In short:

1. Read your assigned issue
2. Branch: `feature/<issue-number>-<description>`
3. Code, test, push
4. Open a PR linking the issue and wiki documentation
5. Address review, merge, delete branch

## Citation

```bibtex
@article{xu2024sen2fire,
  title={Sen2Fire: A Challenging Benchmark Dataset for Wildfire Detection using Sentinel Data},
  author={Xu, Yonghao and Berg, Amanda and Haglund, Leif},
  journal={arXiv preprint arXiv:2403.17884},
  year={2024}
}
```
