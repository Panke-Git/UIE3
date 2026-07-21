"""Strict TSV-backed paired image dataset for the LSUI baseline."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ManifestEntry:
    """One exact-stem input/GT pair from a three-column manifest."""

    sample_id: str
    input_relative_path: str
    gt_relative_path: str
    input_path: Path
    gt_path: Path


def _resolve_dataset_file(dataset_root: Path, relative_text: str, line_number: int) -> Path:
    if not relative_text:
        raise ValueError(f"Manifest line {line_number} contains an empty image path.")
    relative_path = Path(relative_text)
    if relative_path.is_absolute():
        raise ValueError(
            f"Manifest line {line_number} must use a relative path, got {relative_text!r}."
        )
    resolved = (dataset_root / relative_path).resolve(strict=False)
    try:
        resolved.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(
            f"Manifest line {line_number} escapes dataset_root: {relative_text!r}."
        ) from exc
    if not resolved.exists():
        raise FileNotFoundError(
            f"Manifest line {line_number} references a missing file: {resolved}"
        )
    if not resolved.is_file():
        raise ValueError(
            f"Manifest line {line_number} does not reference a regular file: {resolved}"
        )
    return resolved


def _read_manifest(dataset_root: Path, manifest_path: Path) -> List[ManifestEntry]:
    if not manifest_path.exists() or not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist or is not a file: {manifest_path}")

    entries: List[ManifestEntry] = []
    sample_ids = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            columns = line.split("\t")
            if len(columns) != 3:
                raise ValueError(
                    f"Manifest line {line_number} must contain exactly 3 tab-separated "
                    f"columns; found {len(columns)}."
                )
            sample_id, input_relative_path, gt_relative_path = columns
            if not sample_id or not sample_id.strip():
                raise ValueError(f"Manifest line {line_number} contains an empty sample_id.")
            if sample_id in sample_ids:
                raise ValueError(
                    f"Manifest contains duplicate sample_id {sample_id!r} at line {line_number}."
                )
            sample_ids.add(sample_id)
            entries.append(
                ManifestEntry(
                    sample_id=sample_id,
                    input_relative_path=input_relative_path,
                    gt_relative_path=gt_relative_path,
                    input_path=_resolve_dataset_file(
                        dataset_root, input_relative_path, line_number
                    ),
                    gt_path=_resolve_dataset_file(
                        dataset_root, gt_relative_path, line_number
                    ),
                )
            )
    if not entries:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return entries


def _load_rgb_array(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to decode image as RGB: {path}: {exc}") from exc
    return array / np.float32(255.0)


def _paired_reflect_pad(
    input_array: np.ndarray,
    target_array: np.ndarray,
    pad_height: int,
    pad_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    padding = ((0, pad_height), (0, pad_width), (0, 0))
    try:
        return (
            np.pad(input_array, padding, mode="reflect"),
            np.pad(target_array, padding, mode="reflect"),
        )
    except ValueError as exc:
        raise ValueError(
            "Reflect padding failed; the source image may be too small for the "
            f"requested padding (bottom={pad_height}, right={pad_width})."
        ) from exc


class PairedImageDataset(Dataset):
    """Load paired RGB tensors from a strict root-relative TSV manifest.

    Random crop and augmentation are enabled only when ``training=True``.
    The same sampled spatial operation is always applied to input and target.
    """

    def __init__(
        self,
        dataset_root: str,
        manifest_path: str,
        *,
        training: bool,
        patch_size: Optional[int] = None,
        enable_hflip: bool = False,
        enable_vflip: bool = False,
        enable_rot90: bool = False,
        pad_if_smaller: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root).expanduser().resolve(strict=False)
        if not self.dataset_root.exists() or not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"dataset_root does not exist or is not a directory: {self.dataset_root}"
            )
        self.manifest_path = Path(manifest_path).expanduser().resolve(strict=False)
        if patch_size is not None and (type(patch_size) is not int or patch_size <= 0):
            raise ValueError(f"patch_size must be a positive integer or None, got {patch_size!r}.")
        self.training = bool(training)
        self.patch_size = patch_size
        self.enable_hflip = bool(enable_hflip)
        self.enable_vflip = bool(enable_vflip)
        self.enable_rot90 = bool(enable_rot90)
        self.pad_if_smaller = bool(pad_if_smaller)
        self.entries = _read_manifest(self.dataset_root, self.manifest_path)

    def __len__(self) -> int:
        return len(self.entries)

    def _training_transform(
        self, input_array: np.ndarray, target_array: np.ndarray, sample_id: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.patch_size is not None:
            patch_size = self.patch_size
            height, width = input_array.shape[:2]
            pad_height = max(0, patch_size - height)
            pad_width = max(0, patch_size - width)
            if pad_height or pad_width:
                if not self.pad_if_smaller:
                    raise ValueError(
                        f"Sample {sample_id!r} has size {height}x{width}, smaller than "
                        f"patch_size={patch_size}, and pad_if_smaller is false."
                    )
                input_array, target_array = _paired_reflect_pad(
                    input_array, target_array, pad_height, pad_width
                )
                height, width = input_array.shape[:2]
            top = random.randint(0, height - patch_size)
            left = random.randint(0, width - patch_size)
            input_array = input_array[top : top + patch_size, left : left + patch_size]
            target_array = target_array[top : top + patch_size, left : left + patch_size]

        if self.enable_hflip and random.random() < 0.5:
            input_array = np.flip(input_array, axis=1)
            target_array = np.flip(target_array, axis=1)
        if self.enable_vflip and random.random() < 0.5:
            input_array = np.flip(input_array, axis=0)
            target_array = np.flip(target_array, axis=0)
        if self.enable_rot90:
            rotations = random.randint(0, 3)
            if rotations:
                input_array = np.rot90(input_array, rotations, axes=(0, 1))
                target_array = np.rot90(target_array, rotations, axes=(0, 1))
        return input_array, target_array

    @staticmethod
    def _to_tensor(array: np.ndarray) -> torch.Tensor:
        contiguous = np.ascontiguousarray(array, dtype=np.float32)
        tensor = torch.from_numpy(contiguous).permute(2, 0, 1).contiguous()
        return tensor

    def __getitem__(self, index: int) -> Dict[str, object]:
        entry = self.entries[index]
        input_array = _load_rgb_array(entry.input_path)
        target_array = _load_rgb_array(entry.gt_path)
        if input_array.shape != target_array.shape:
            raise ValueError(
                f"Input/GT size mismatch for sample {entry.sample_id!r}: "
                f"input={input_array.shape[:2]}, GT={target_array.shape[:2]}."
            )
        if self.training:
            input_array, target_array = self._training_transform(
                input_array, target_array, entry.sample_id
            )
        input_tensor = self._to_tensor(input_array)
        target_tensor = self._to_tensor(target_array)
        if input_tensor.shape[0] != 3 or target_tensor.shape[0] != 3:
            raise RuntimeError(f"RGB conversion failed for sample {entry.sample_id!r}.")
        return {
            "sample_id": entry.sample_id,
            "input": input_tensor,
            "target": target_tensor,
            "input_relative_path": entry.input_relative_path,
            "gt_relative_path": entry.gt_relative_path,
        }
