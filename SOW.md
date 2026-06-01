# Standard Operating Workflow (SOW)

## 1. Project Overview

This project develops a hybrid quantum-classical machine learning model for wildfire detection using Sentinel-2 satellite imagery. The model performs pixel-level semantic segmentation on the Sen2Fire dataset, comparing classical U-Net architectures against quantum-enhanced variants.

**Tech stack**: PyTorch + PennyLane (quantum), SLURM HPC for training.

## 2. Codebase Structure

```
wildfire-detection-2026/
├── configs/                          # Experiment configuration files (YAML)
│   ├── classical_baseline.yaml       # Classical U-Net, mode 5 (SWIR + aerosol)
│   └── quantum_strongly_entangling.yaml
│
├── models/                           # Model architectures (swappable)
│   ├── __init__.py                   # MODEL_REGISTRY + get_model()
│   ├── unet.py                       # Classical U-Net
│   └── quantum_unet.py              # Quantum-enhanced U-Net
│
├── circuits/                         # Quantum circuits (swappable)
│   ├── __init__.py                   # CIRCUIT_REGISTRY + get_circuit()
│   ├── strongly_entangling.py        # AmplitudeEmbedding + StronglyEntanglingLayers
│   └── ry_cnot.py                    # RY encoding + CNOT entanglement
│
├── losses/                           # Loss functions (swappable)
│   ├── __init__.py                   # LOSS_REGISTRY + get_loss()
│   └── cross_entropy.py              # Weighted cross-entropy (current default)
│
├── dataset/
│   ├── __init__.py
│   ├── sen2fire.py                   # PyTorch dataset class
│   ├── train.txt / val.txt / test.txt
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py                    # Evaluation metrics (OA, F1, IoU, etc.)
│   └── visualization.py             # Prediction maps, overlays
│
├── experiments/                      # Results from training/test runs
│   ├── classical_baseline/           # Existing classical results
│   │   ├── config.yaml               # Config snapshot
│   │   ├── Training_log.txt
│   │   ├── training_history.npz
│   │   ├── training_plot.png
│   │   └── maps/                     # Scene prediction maps
│   └── <experiment_name>_<MMDD_HHMM>/
│
├── notebooks/                        # Shared reference notebooks
│   └── data_exploration.ipynb
│
├── scripts/                          # SLURM job submission scripts
│   ├── submit_train.sh
│   └── submit_test.sh
│
├── .github/
│   ├── pull_request_template.md
│   └── ISSUE_TEMPLATE/
│
├── train.py                          # Training entry point (config-driven)
├── test.py                           # Inference entry point (config-driven)
├── requirements.txt
├── SOW.md                            # This document
└── README.md
```

### Key design principles

- **One entry point each** for training (`train.py`) and inference (`test.py`), driven by YAML config files.
- **Registry pattern** for models, circuits, and losses. Adding a new variant means adding a new file and one line in the registry — no changes to `train.py` or `test.py`.
- **Config files** fully describe an experiment. Every run copies its config into the output directory for reproducibility.
- **Experiment results** are committed (except `.pth` checkpoints, which are gitignored due to size). Each PR that involves a training run should include the best run's results.

## 3. Setup

```bash
# Clone the repo
git clone git@git.appdevclub.com:IonQSPRING2026/wildfire-detection-2026.git
cd wildfire-detection-2026

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from models import get_model; print('OK')"
```

The Sen2Fire dataset should be placed one level up from the repo root at `../Sen2Fire/`. See `README.md` for the expected directory layout.

## 4. Working on a Task

### Step 1: Read the issue

A lead creates a Gitea issue for each task and assigns it to you. Read the issue description carefully to understand the scope, acceptance criteria, and which area of the codebase is affected.

### Step 2: Research (if applicable)

Some tasks require research before implementation (e.g., "research alternative loss functions"). For research tasks:

1. Do the research.
2. Document your findings in the **Gitea Wiki** under a clearly named page.
3. Update the issue with a summary and link to the wiki page.
4. Discuss with a lead whether a follow-up implementation task is needed.

Research-only tasks do **not** produce a PR. The deliverable is the wiki page.

### Step 3: Create a feature branch

```bash
git checkout master
git pull origin master
git checkout -b feature/<issue-number>-<short-description>
```

**Branch naming convention**: `feature/<issue-number>-<short-description>`

Examples:
- `feature/12-cli-args`
- `feature/15-focal-loss`
- `feature/18-heatmap-visualization`
- `feature/21-qufex-circuit`

### Step 4: Do the work

Follow the coding guidelines in Section 6. Key rules:

- **Pipeline improvements** (CLI args, new metrics, visualizations): modify existing files (`train.py`, `utils/metrics.py`, etc.).
- **New approaches** (new model, circuit, loss function): create a **new file** in the appropriate directory and register it. Do not modify existing model/circuit/loss files.
- **Training runs**: create a new config YAML in `configs/`, run the experiment, and include the best run's results directory under `experiments/`.

### Step 5: Test your changes

Before pushing, verify:
- Your code runs without errors (at minimum, test with a small batch locally or on the HPC).
- Existing functionality is not broken (other configs still work).
- If you added a new registered component, verify it can be instantiated:
  ```bash
  python -c "from models import get_model; m = get_model({'model': 'your_model', 'mode': 5, 'n_classes': 2})"
  ```

### Step 6: Push and open a PR

```bash
git push -u origin feature/<issue-number>-<short-description>
```

Then open a Pull Request on Gitea. Fill out the PR template completely — it's short and takes under 2 minutes. Make sure to:

1. **Link the issue**: Write `Closes #<number>` in the PR description.
2. **Link the wiki**: If you wrote documentation, include the wiki page URL.
3. **Include experiment results**: If your task involved a training run, mention which config you used and summarize the key metrics.

### Step 7: Address review feedback

A lead or a peer will review your PR. Address any requested changes by pushing additional commits to the same branch. Do not force-push.

### Step 8: Merge

Once approved, the PR is merged into `master` and the feature branch is deleted.

## 5. How to Add New Components

### Adding a new model

1. Create `models/my_new_model.py` with a class that has the signature:
   ```python
   class MyNewModel(nn.Module):
       def __init__(self, n_classes, n_channels, config=None):
   ```
2. Register it in `models/__init__.py`:
   ```python
   from models.my_new_model import MyNewModel
   MODEL_REGISTRY['my_new_model'] = MyNewModel
   ```
3. Create a config file `configs/my_new_model.yaml`:
   ```yaml
   model: my_new_model
   mode: 5
   # ... other params
   ```
4. Run: `python train.py --config configs/my_new_model.yaml`

### Adding a new quantum circuit

1. Create `circuits/my_circuit.py` with a factory function:
   ```python
   def create_my_circuit(config):
       n_qubits = config.get('n_qubits', 8)
       n_layers = config.get('n_layers', 2)
       dev = qml.device('default.qubit', wires=n_qubits)

       @qml.qnode(dev, interface='torch', diff_method='adjoint')
       def circuit(inputs, weights):
           # Your circuit here
           ...
           return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

       weight_shapes = {"weights": (...)}  # Shape of trainable weights
       return circuit, weight_shapes
   ```
2. Register it in `circuits/__init__.py`:
   ```python
   from circuits.my_circuit import create_my_circuit
   CIRCUIT_REGISTRY['my_circuit'] = create_my_circuit
   ```
3. Reference it in a config:
   ```yaml
   model: quantum_unet
   circuit: my_circuit
   ```

### Adding a new loss function

1. Create `losses/my_loss.py`:
   ```python
   def create_my_loss(config):
       return MyLossClass(...)
   ```
2. Register in `losses/__init__.py`:
   ```python
   from losses.my_loss import create_my_loss
   LOSS_REGISTRY['my_loss'] = create_my_loss
   ```
3. Reference in config:
   ```yaml
   loss: my_loss
   ```

## 6. Coding Guidelines

- **Style**: PEP 8, 4-space indentation.
- **Imports**: standard library, then third-party, then local — separated by blank lines.
- **No hardcoded paths**. Use config files or CLI arguments.
- **No HPC-specific imports** in Python source files (e.g., `envmodules`). Environment setup belongs in SLURM scripts.
- **Comments**: add them where the logic isn't obvious. Don't over-comment trivial code.
- **Naming**: snake_case for functions/variables, PascalCase for classes.

## 7. Running Experiments

### Locally (small test)

```bash
# Train with a config
python train.py --config configs/classical_baseline.yaml --epochs 1 --batch_size 2

# Run inference
python test.py --config configs/classical_baseline.yaml \
    --restore_from experiments/<run_dir>/best_model.pth
```

### On the HPC (full run)

```bash
# Submit training
sbatch scripts/submit_train.sh --config configs/your_config.yaml

# Submit inference
sbatch scripts/submit_test.sh --config configs/your_config.yaml \
    --restore_from experiments/<run_dir>/best_model.pth
```

### Experiment output

Every training run creates:
```
experiments/<experiment_name>_<MMDD_HHMM>/
├── config.yaml           # Exact config used (for reproducibility)
├── Training_log.txt      # Epoch-by-epoch metrics
├── training_history.npz  # Raw training history arrays
├── training_plot.png     # Loss/accuracy curves
└── best_model.pth        # Best checkpoint (gitignored)
```

When committing experiment results, include everything **except** `.pth` files (they're ~66MB each and gitignored). Mention the key metrics (F1, IoU, OA) in your PR description so reviewers can compare.

#### Model Weight Storage Protocol
Because `.pth` files are too large for standard GitHub tracking, we store them externally in Google Drive. Whenever a new experiment run is completed and deemed useful, the author must:
1. Rename the local `.pth` file to include the experiment directory name to avoid naming collisions: `<experiment_name>_<MMDD_HHMM>_best_model.pth`.
2. Upload the renamed file to the shared `QFireNet_Weights` Google Drive folder.
3. Include the Google Drive link to the uploaded weight file in your PR description.
## 8. Notebooks

Notebooks are gitignored by default to prevent accidental commits of large files with output cells. For local experimentation, create notebooks anywhere — they won't be tracked.

**Shared reference notebooks** live in `notebooks/` and are committed explicitly:
```bash
git add -f notebooks/my_useful_notebook.ipynb
```

Only commit a notebook if it provides lasting value to the team (data exploration, result analysis, demos). Discuss with a lead first.

You can import the project's modules directly in notebooks:
```python
from models import get_model
from dataset.sen2fire import Sen2FireDataSet
from utils.metrics import eval_image
```

## 9. Quick Reference

| What you want to do | Where to look |
|---|---|
| Change hyperparameters for a run | Create/edit a config YAML in `configs/` |
| Add a new model architecture | `models/` directory + registry |
| Add a new quantum circuit | `circuits/` directory + registry |
| Add a new loss function | `losses/` directory + registry |
| Add new evaluation metrics | `utils/metrics.py` |
| Add new visualizations | `utils/visualization.py` |
| Change data preprocessing | `dataset/sen2fire.py` |
| Modify the training loop | `train.py` |
| Modify inference/evaluation | `test.py` |
| Change SLURM job settings | `scripts/submit_train.sh` or `scripts/submit_test.sh` |
| Find past experiment results | `experiments/` directory |
| Find research documentation | Gitea Wiki |
