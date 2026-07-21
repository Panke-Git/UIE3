"""Runtime tests for the strict paired-image manifest dataset."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.paired_image_dataset import PairedImageDataset


def _save_image(path: Path, array: np.ndarray, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode=mode).save(path)


def _make_manifest(root: Path, manifest: Path, rows: list) -> None:
    manifest.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")


def _rgb_pattern(height: int, width: int) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    return np.stack(
        ((x * 17 + y * 3) % 256, (x * 5 + y * 19) % 256, (x + y * 7) % 256),
        axis=-1,
    ).astype(np.uint8)


def test_normal_manifest_rgb_conversion_and_range(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    gray = np.arange(16 * 18, dtype=np.uint8).reshape(16, 18)
    rgba = np.concatenate(
        [_rgb_pattern(16, 18), np.full((16, 18, 1), 255, dtype=np.uint8)], axis=-1
    )
    _save_image(root / "Train/input/a.png", gray, mode="L")
    _save_image(root / "Train/GT/a.png", rgba, mode="RGBA")
    manifest = tmp_path / "manifest.tsv"
    _make_manifest(
        root,
        manifest,
        [("a", "Train/input/a.png", "Train/GT/a.png")],
    )
    dataset = PairedImageDataset(str(root), str(manifest), training=False)
    sample = dataset[0]
    assert sample["sample_id"] == "a"
    assert sample["input"].dtype == torch.float32
    assert sample["target"].dtype == torch.float32
    assert sample["input"].shape == (3, 16, 18)
    assert sample["target"].shape == (3, 16, 18)
    assert 0.0 <= float(sample["input"].min()) <= float(sample["input"].max()) <= 1.0
    assert 0.0 <= float(sample["target"].min()) <= float(sample["target"].max()) <= 1.0


def test_duplicate_sample_id_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image = _rgb_pattern(12, 12)
    for name in ("a", "b"):
        _save_image(root / f"input/{name}.png", image)
        _save_image(root / f"gt/{name}.png", image)
    manifest = tmp_path / "manifest.tsv"
    _make_manifest(
        root,
        manifest,
        [
            ("duplicate", "input/a.png", "gt/a.png"),
            ("duplicate", "input/b.png", "gt/b.png"),
        ],
    )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        PairedImageDataset(str(root), str(manifest), training=False)


@pytest.mark.parametrize("line", ["a\tinput.png\n", "a\tinput.png\tgt.png\textra\n"])
def test_invalid_manifest_column_count_is_rejected(tmp_path: Path, line: str) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 3"):
        PairedImageDataset(str(root), str(manifest), training=False)


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    image = _rgb_pattern(12, 12)
    _save_image(tmp_path / "outside.png", image)
    _save_image(root / "gt.png", image)
    manifest = tmp_path / "manifest.tsv"
    _make_manifest(root, manifest, [("a", "../outside.png", "gt.png")])
    with pytest.raises(ValueError, match="escapes dataset_root"):
        PairedImageDataset(str(root), str(manifest), training=False)


def test_input_gt_size_mismatch_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _save_image(root / "input.png", _rgb_pattern(12, 13))
    _save_image(root / "gt.png", _rgb_pattern(12, 14))
    manifest = tmp_path / "manifest.tsv"
    _make_manifest(root, manifest, [("a", "input.png", "gt.png")])
    dataset = PairedImageDataset(str(root), str(manifest), training=False)
    with pytest.raises(ValueError, match="size mismatch"):
        _ = dataset[0]


def test_paired_crop_and_augmentation_are_spatially_identical(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image = _rgb_pattern(28, 31)
    _save_image(root / "input.png", image)
    _save_image(root / "gt.png", image)
    manifest = tmp_path / "manifest.tsv"
    _make_manifest(root, manifest, [("paired", "input.png", "gt.png")])
    dataset = PairedImageDataset(
        str(root),
        str(manifest),
        training=True,
        patch_size=16,
        enable_hflip=True,
        enable_vflip=True,
        enable_rot90=True,
    )
    sample = dataset[0]
    assert sample["input"].shape == (3, 16, 16)
    assert torch.equal(sample["input"], sample["target"])


def test_pad_if_smaller_uses_paired_reflect_padding(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image = _rgb_pattern(7, 9)
    _save_image(root / "input.png", image)
    _save_image(root / "gt.png", image)
    manifest = tmp_path / "manifest.tsv"
    _make_manifest(root, manifest, [("small", "input.png", "gt.png")])
    padded = PairedImageDataset(
        str(root),
        str(manifest),
        training=True,
        patch_size=13,
        pad_if_smaller=True,
    )[0]
    assert padded["input"].shape == (3, 13, 13)
    assert torch.equal(padded["input"], padded["target"])

    unpadded_dataset = PairedImageDataset(
        str(root),
        str(manifest),
        training=True,
        patch_size=13,
        pad_if_smaller=False,
    )
    with pytest.raises(ValueError, match="pad_if_smaller is false"):
        _ = unpadded_dataset[0]


def test_validation_ignores_crop_and_random_augmentation(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image = _rgb_pattern(14, 17)
    _save_image(root / "input.png", image)
    _save_image(root / "gt.png", image)
    manifest = tmp_path / "manifest.tsv"
    _make_manifest(root, manifest, [("validation", "input.png", "gt.png")])
    sample = PairedImageDataset(
        str(root),
        str(manifest),
        training=False,
        patch_size=8,
        enable_hflip=True,
        enable_vflip=True,
        enable_rot90=True,
    )[0]
    assert sample["input"].shape == (3, 14, 17)
    assert torch.equal(sample["input"], sample["target"])
