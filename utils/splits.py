"""utils/splits.py — dataset splitting utilities.

Provides :func:`make_random_split_lists`, which pools all patch filenames from
the scene-based split files defined in a training config, shuffles them
deterministically, and re-partitions them into train / val / test by ratio.
This removes the train↔test domain shift caused by the original scene-level
partitioning (train=scenes1-2, val=scene3, test=scene4).
"""
import io
import os

import numpy as np


def make_random_split_lists(config, seed):
    """Pool all patches from the scene-based split files, shuffle once, and
    re-partition into train / val / test by ratio.

    Returns a tuple ``(split_ios, split_sizes)`` where:

    * ``split_ios`` is a dict mapping ``'train'``/``'val'``/``'test'`` to an
      :class:`io.StringIO` containing the newline-delimited patch paths for
      that split.  These can be passed directly to ``_InMemoryDataSet`` in
      place of the original ``.txt`` file paths.
    * ``split_sizes`` is a dict mapping the same keys to patch counts.

    Config keys consumed (all optional):
        ``train_list``              path to original train list file
        ``val_list``                path to original val list file
        ``test_list``               path to original test list file
        ``random_split_train_ratio``  float, default 0.70
        ``random_split_val_ratio``    float, default 0.15
        (test ratio is inferred as the remainder)

    Parameters
    ----------
    config : dict
        Experiment configuration dictionary (as loaded by ``load_config``).
    seed : int
        RNG seed for reproducible shuffling.
    """
    # Gather every patch path mentioned in any of the three list files.
    all_patches = []
    for list_key in ('train_list', 'val_list', 'test_list'):
        list_path = config.get(list_key)
        if list_path and os.path.isfile(list_path):
            with open(list_path) as fh:
                all_patches.extend(line.strip() for line in fh if line.strip())

    # Deduplicate while preserving order, then shuffle deterministically.
    seen = set()
    unique_patches = []
    for p in all_patches:
        if p not in seen:
            seen.add(p)
            unique_patches.append(p)

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_patches)
    n = len(unique_patches)

    train_ratio = config.get('random_split_train_ratio', 0.70)
    val_ratio   = config.get('random_split_val_ratio',   0.15)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = {
        'train': unique_patches[:n_train],
        'val':   unique_patches[n_train:n_train + n_val],
        'test':  unique_patches[n_train + n_val:],
    }

    split_ios = {
        k: io.StringIO('\n'.join(v))
        for k, v in splits.items()
    }
    split_sizes = {k: len(v) for k, v in splits.items()}
    return split_ios, split_sizes
