#!/usr/bin/env python3
"""Convert CaBuAr Hugging Face HDF5 files into Sen2Fire-style NPZ patches.

Output patches contain the keys expected by dataset/sen2fire.py:
image: 12-band Sentinel-2 patch in (C, H, W)
label: binary mask in (H, W)
aerosol: all-zero placeholder in (H, W), because CaBuAr has no aerosol band
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import hdf5plugin  # noqa: F401 - registers HDF5 compression filters
import h5py
import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm


DEFAULT_TRAIN_SPLITS = ("0", "1", "2")
DEFAULT_VAL_SPLITS = ("3",)
DEFAULT_TEST_SPLITS = ("4",)
DEFAULT_HDF5_FILE = "raw/patched/512x512.hdf5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CaBuAr and convert it to this repo's NPZ patch format."
    )
    parser.add_argument(
        "--output_dir",
        default="cabuar_preprocessing/output",
        help="Directory for converted patches and split files.",
    )
    parser.add_argument(
        "--dataset_name",
        default="DarthReca/california_burned_areas",
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--hdf5_path",
        default=None,
        help="Optional local CaBuAr HDF5 file. If omitted, downloads from Hugging Face.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Accepted for old commands; no longer needed because this script reads HDF5 directly.",
    )
    parser.add_argument(
        "--max_per_split",
        type=int,
        default=None,
        help="Optional cap for quick smoke conversions.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def image_to_chw(post_fire: np.ndarray) -> np.ndarray:
    image = np.asarray(post_fire)
    if image.shape == (512, 512, 12):
        image = np.transpose(image, (2, 0, 1))
    elif image.shape != (12, 512, 512):
        raise ValueError(f"Expected post_fire shape (512, 512, 12), got {image.shape}")
    return image.astype(np.float32, copy=False)


def mask_to_label(mask: np.ndarray) -> np.ndarray:
    label = np.asarray(mask)
    if label.shape == (512, 512, 1):
        label = label[:, :, 0]
    elif label.shape == (1, 512, 512):
        label = label[0]
    elif label.shape != (512, 512):
        raise ValueError(f"Expected mask shape (512, 512, 1), got {label.shape}")
    return (label > 0).astype(np.uint8)


def normalize_fold(value) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def resolve_hdf5_path(args: argparse.Namespace) -> Path:
    if args.hdf5_path is not None:
        return Path(args.hdf5_path)

    downloaded = hf_hub_download(
        repo_id=args.dataset_name,
        repo_type="dataset",
        filename=DEFAULT_HDF5_FILE,
    )
    return Path(downloaded)


def convert_split(
    h5_file: h5py.File,
    split_name: str,
    patch_root: Path,
    max_per_split: int | None,
) -> list[str]:
    split_dir = patch_root / f"fold_{split_name}"
    split_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    split_keys = [
        key for key, values in h5_file.items()
        if normalize_fold(values.attrs["fold"]) == split_name
    ]
    total = len(split_keys) if max_per_split is None else min(len(split_keys), max_per_split)

    for index, key in enumerate(tqdm(split_keys, desc=f"Converting split {split_name}", total=total)):
        if max_per_split is not None and index >= max_per_split:
            break

        values = h5_file[key]
        image = image_to_chw(values["post_fire"][...])
        label = mask_to_label(values["mask"][...])
        aerosol = np.zeros(label.shape, dtype=np.float32)

        filename = f"{safe_name(split_name)}_{safe_name(key)}.npz"
        patch_path = split_dir / filename
        np.savez_compressed(patch_path, image=image, label=label, aerosol=aerosol)
        entries.append(str(patch_path.relative_to(patch_root)))

    return entries


def write_split_file(path: Path, entries: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(entries) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    data_root = output_dir / "cabuar_sen2fire"
    split_root = output_dir / "splits"
    hdf5_path = resolve_hdf5_path(args)

    split_groups = {
        "train": DEFAULT_TRAIN_SPLITS,
        "val": DEFAULT_VAL_SPLITS,
        "test": DEFAULT_TEST_SPLITS,
    }

    summary = {}
    with h5py.File(hdf5_path, "r") as h5_file:
        for output_split, source_splits in split_groups.items():
            entries = []
            for source_split in source_splits:
                entries.extend(
                    convert_split(
                        h5_file,
                        source_split,
                        data_root,
                        args.max_per_split,
                    )
                )
            write_split_file(split_root / f"{output_split}.txt", entries)
            summary[output_split] = len(entries)

    print("Converted CaBuAr to Sen2Fire-style NPZ patches:")
    print(f"  source:   {hdf5_path}")
    print(f"  data_dir: {data_root}")
    print(f"  train: {summary['train']} patches")
    print(f"  val:   {summary['val']} patches")
    print(f"  test:  {summary['test']} patches")
    print("  config: configs/classical_cabuar.yaml")


if __name__ == "__main__":
    main()
