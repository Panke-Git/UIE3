#!/usr/bin/env python3
"""Audit an LSUI19 paired dataset and generate deterministic split manifests.

The dataset is treated as read-only. Manifests are generated only after every
blocking audit check passes. This module intentionally depends only on the
Python standard library and Pillow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import traceback
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DATASET_IDENTIFIER = "LSUI19"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED_TRAIN_POOL_COUNT = 3851
EXPECTED_TEST_COUNT = 428
DEFAULT_SPLIT_SEED = 3407
DEFAULT_VALIDATION_COUNT = 385
MANIFEST_FILENAMES = (
    "train.tsv",
    "validation.tsv",
    "test.tsv",
    "SPLIT_METADATA.json",
)
SOURCE_DIRECTORIES = (
    ("train_input", Path("Train/input")),
    ("train_gt", Path("Train/GT")),
    ("val_input", Path("Val/input")),
    ("val_gt", Path("Val/GT")),
)


class UsageError(Exception):
    """An argument, path, or runtime dependency is invalid."""


def utc_timestamp() -> str:
    """Return a stable, timezone-explicit UTC timestamp."""

    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def natural_sort_key(value: str) -> Tuple[Tuple[int, Any], ...]:
    """Return a deterministic, case-aware natural-sort key."""

    parts = re.split(r"(\d+)", value)
    key: List[Tuple[int, Any]] = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.casefold(), part))
    return tuple(key)


def natural_sorted(values: Iterable[str]) -> List[str]:
    return sorted(values, key=natural_sort_key)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def safe_error(exc: BaseException) -> str:
    """Format an exception without hiding its type."""

    return f"{type(exc).__name__}: {exc}"


@dataclass
class ImageRecord:
    path: Path
    relative_path: str
    sample_id: str
    extension: str
    size_bytes: int
    readable: bool = False
    read_error: Optional[str] = None
    mode: Optional[str] = None
    mode_group: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    rgb_convertible: bool = False
    rgb_conversion_error: Optional[str] = None
    sha256: Optional[str] = None
    hash_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "relative_path": self.relative_path,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "readable": self.readable,
            "mode": self.mode,
            "mode_group": self.mode_group,
            "size": (
                [self.width, self.height]
                if self.width is not None and self.height is not None
                else None
            ),
            "rgb_convertible": self.rgb_convertible,
        }
        if self.read_error is not None:
            result["read_error"] = self.read_error
        if self.rgb_conversion_error is not None:
            result["rgb_conversion_error"] = self.rgb_conversion_error
        if self.sha256 is not None:
            result["sha256"] = self.sha256
        if self.hash_error is not None:
            result["hash_error"] = self.hash_error
        return result


@dataclass
class DirectoryScan:
    label: str
    path: Path
    relative_directory: str
    records: List[ImageRecord] = field(default_factory=list)
    non_image_files: List[Dict[str, Any]] = field(default_factory=list)
    empty_files: List[str] = field(default_factory=list)
    symlinks: List[str] = field(default_factory=list)
    scan_errors: List[Dict[str, str]] = field(default_factory=list)
    inventory: List[Tuple[str, int, str]] = field(default_factory=list)

    def records_by_stem(self) -> Dict[str, List[ImageRecord]]:
        grouped: Dict[str, List[ImageRecord]] = defaultdict(list)
        for record in self.records:
            grouped[record.sample_id].append(record)
        return dict(grouped)

    def duplicate_stems(self) -> Dict[str, List[str]]:
        return {
            stem: natural_sorted(record.relative_path for record in records)
            for stem, records in self.records_by_stem().items()
            if len(records) > 1
        }

    def unique_records(self) -> Dict[str, ImageRecord]:
        return {
            stem: records[0]
            for stem, records in self.records_by_stem().items()
            if len(records) == 1
        }

    def to_dict(self) -> Dict[str, Any]:
        extension_distribution = Counter(record.extension for record in self.records)
        mode_distribution = Counter(
            record.mode for record in self.records if record.mode is not None
        )
        mode_group_distribution = Counter(
            record.mode_group
            for record in self.records
            if record.mode_group is not None
        )
        size_distribution = Counter(
            f"{record.width}x{record.height}"
            for record in self.records
            if record.width is not None and record.height is not None
        )
        unreadable = [
            {
                "sample_id": record.sample_id,
                "relative_path": record.relative_path,
                "error": record.read_error,
            }
            for record in self.records
            if not record.readable
        ]
        rgb_failures = [
            {
                "sample_id": record.sample_id,
                "relative_path": record.relative_path,
                "mode": record.mode,
                "error": record.rgb_conversion_error,
            }
            for record in self.records
            if record.readable and not record.rgb_convertible
        ]
        hash_failures = [
            {
                "sample_id": record.sample_id,
                "relative_path": record.relative_path,
                "error": record.hash_error,
            }
            for record in self.records
            if record.hash_error is not None
        ]
        return {
            "directory": self.relative_directory,
            "image_count": len(self.records),
            "extension_distribution": dict(sorted(extension_distribution.items())),
            "mode_distribution": dict(sorted(mode_distribution.items())),
            "mode_group_distribution": dict(sorted(mode_group_distribution.items())),
            "size_distribution": dict(
                sorted(size_distribution.items(), key=lambda item: natural_sort_key(item[0]))
            ),
            "duplicate_stems": dict(
                sorted(self.duplicate_stems().items(), key=lambda item: natural_sort_key(item[0]))
            ),
            "unreadable_or_corrupt_images": unreadable,
            "rgb_conversion_failures": rgb_failures,
            "hash_failures": hash_failures,
            "empty_files": natural_sorted(self.empty_files),
            "non_image_file_count": len(self.non_image_files),
            "non_image_files": sorted(
                self.non_image_files,
                key=lambda item: natural_sort_key(str(item["relative_path"])),
            ),
            "symlinks": natural_sorted(self.symlinks),
            "scan_errors": self.scan_errors,
            "images": sorted(
                (record.to_dict() for record in self.records),
                key=lambda item: natural_sort_key(str(item["relative_path"])),
            ),
        }


def load_pillow() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise UsageError(
            "Pillow is required for image validation but is not installed. "
            "Install it outside this script before running the audit."
        ) from exc
    return Image


def inspect_image(record: ImageRecord, image_module: Any) -> None:
    """Verify, fully decode, and test RGB conversion without saving the image."""

    try:
        with warnings.catch_warnings():
            decompression_warning = getattr(
                image_module, "DecompressionBombWarning", Warning
            )
            warnings.simplefilter("error", decompression_warning)
            with image_module.open(record.path) as image:
                image.verify()
            with image_module.open(record.path) as image:
                image.load()
                record.mode = str(image.mode)
                record.mode_group = (
                    record.mode if record.mode in {"RGB", "L", "RGBA"} else "OTHER"
                )
                record.width, record.height = image.size
                try:
                    with image.convert("RGB") as converted:
                        converted.load()
                    record.rgb_convertible = True
                except Exception as exc:
                    record.rgb_conversion_error = safe_error(exc)
        record.readable = True
    except Exception as exc:
        record.read_error = safe_error(exc)
        record.readable = False


def scan_directory(
    label: str,
    directory: Path,
    root: Path,
    image_module: Any,
    hash_files: bool,
) -> DirectoryScan:
    scan = DirectoryScan(
        label=label,
        path=directory,
        relative_directory=directory.relative_to(root).as_posix(),
    )

    def on_walk_error(exc: OSError) -> None:
        scan.scan_errors.append(
            {"path": str(getattr(exc, "filename", directory)), "error": safe_error(exc)}
        )

    for current, directory_names, file_names in os.walk(
        directory, topdown=True, followlinks=False, onerror=on_walk_error
    ):
        current_path = Path(current)
        directory_names.sort(key=natural_sort_key)
        file_names.sort(key=natural_sort_key)

        traversable: List[str] = []
        for name in directory_names:
            child = current_path / name
            relative = child.relative_to(root).as_posix()
            if child.is_symlink():
                scan.symlinks.append(relative)
                try:
                    size = child.lstat().st_size
                except OSError:
                    size = -1
                scan.inventory.append((relative, size, "symlink_directory"))
            else:
                traversable.append(name)
        directory_names[:] = traversable

        for name in file_names:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                scan.symlinks.append(relative)
                try:
                    size = path.lstat().st_size
                except OSError:
                    size = -1
                scan.inventory.append((relative, size, "symlink_file"))
                continue

            try:
                size = path.stat().st_size
            except OSError as exc:
                scan.scan_errors.append({"path": relative, "error": safe_error(exc)})
                scan.inventory.append((relative, -1, "stat_error"))
                continue

            extension = path.suffix.lower()
            kind = "image" if extension in SUPPORTED_EXTENSIONS else "non_image"
            scan.inventory.append((relative, size, kind))
            if size == 0:
                scan.empty_files.append(relative)

            if extension not in SUPPORTED_EXTENSIONS:
                scan.non_image_files.append(
                    {"relative_path": relative, "size_bytes": size}
                )
                continue

            record = ImageRecord(
                path=path,
                relative_path=relative,
                sample_id=path.stem,
                extension=extension,
                size_bytes=size,
            )
            inspect_image(record, image_module)
            if hash_files:
                try:
                    record.sha256 = sha256_file(path)
                except Exception as exc:
                    record.hash_error = safe_error(exc)
            scan.records.append(record)

    return scan


def pair_audit(
    input_scan: DirectoryScan, gt_scan: DirectoryScan
) -> Tuple[Dict[str, Any], Dict[str, Tuple[ImageRecord, ImageRecord]]]:
    input_by_stem = input_scan.records_by_stem()
    gt_by_stem = gt_scan.records_by_stem()
    input_stems = set(input_by_stem)
    gt_stems = set(gt_by_stem)
    input_missing_gt = natural_sorted(input_stems - gt_stems)
    gt_missing_input = natural_sorted(gt_stems - input_stems)

    pairs: Dict[str, Tuple[ImageRecord, ImageRecord]] = {}
    size_mismatches: List[Dict[str, Any]] = []
    rgb_conversion_failures: List[Dict[str, Any]] = []
    for sample_id in natural_sorted(input_stems & gt_stems):
        if len(input_by_stem[sample_id]) != 1 or len(gt_by_stem[sample_id]) != 1:
            continue
        input_record = input_by_stem[sample_id][0]
        gt_record = gt_by_stem[sample_id][0]
        pairs[sample_id] = (input_record, gt_record)
        if (
            input_record.readable
            and gt_record.readable
            and (input_record.width, input_record.height)
            != (gt_record.width, gt_record.height)
        ):
            size_mismatches.append(
                {
                    "sample_id": sample_id,
                    "input_relative_path": input_record.relative_path,
                    "gt_relative_path": gt_record.relative_path,
                    "input_size": [input_record.width, input_record.height],
                    "gt_size": [gt_record.width, gt_record.height],
                }
            )
        if not (input_record.rgb_convertible and gt_record.rgb_convertible):
            rgb_conversion_failures.append(
                {
                    "sample_id": sample_id,
                    "input_relative_path": input_record.relative_path,
                    "gt_relative_path": gt_record.relative_path,
                    "input_rgb_convertible": input_record.rgb_convertible,
                    "gt_rgb_convertible": gt_record.rgb_convertible,
                }
            )

    result = {
        "input_count": len(input_scan.records),
        "gt_count": len(gt_scan.records),
        "input_missing_gt": input_missing_gt,
        "gt_missing_input": gt_missing_input,
        "paired_unique_stem_count": len(pairs),
        "pair_size_mismatches": size_mismatches,
        "pair_rgb_conversion_failures": rgb_conversion_failures,
    }
    return result, pairs


def cross_split_hash_matches(
    train_records: Iterable[ImageRecord], val_records: Iterable[ImageRecord]
) -> List[Dict[str, Any]]:
    train_by_hash: Dict[str, List[ImageRecord]] = defaultdict(list)
    val_by_hash: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in train_records:
        if record.sha256 is not None:
            train_by_hash[record.sha256].append(record)
    for record in val_records:
        if record.sha256 is not None:
            val_by_hash[record.sha256].append(record)

    matches: List[Dict[str, Any]] = []
    for digest in sorted(set(train_by_hash) & set(val_by_hash)):
        matches.append(
            {
                "sha256": digest,
                "train": sorted(
                    (
                        {
                            "sample_id": record.sample_id,
                            "relative_path": record.relative_path,
                        }
                        for record in train_by_hash[digest]
                    ),
                    key=lambda item: natural_sort_key(str(item["relative_path"])),
                ),
                "val": sorted(
                    (
                        {
                            "sample_id": record.sample_id,
                            "relative_path": record.relative_path,
                        }
                        for record in val_by_hash[digest]
                    ),
                    key=lambda item: natural_sort_key(str(item["relative_path"])),
                ),
            }
        )
    return matches


def identical_pair_hash_matches(
    train_pairs: Mapping[str, Tuple[ImageRecord, ImageRecord]],
    val_pairs: Mapping[str, Tuple[ImageRecord, ImageRecord]],
) -> List[Dict[str, Any]]:
    train_by_pair_hash: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    val_by_pair_hash: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for sample_id, (input_record, gt_record) in train_pairs.items():
        if input_record.sha256 is not None and gt_record.sha256 is not None:
            train_by_pair_hash[(input_record.sha256, gt_record.sha256)].append(sample_id)
    for sample_id, (input_record, gt_record) in val_pairs.items():
        if input_record.sha256 is not None and gt_record.sha256 is not None:
            val_by_pair_hash[(input_record.sha256, gt_record.sha256)].append(sample_id)

    matches: List[Dict[str, Any]] = []
    for input_hash, gt_hash in sorted(
        set(train_by_pair_hash) & set(val_by_pair_hash)
    ):
        matches.append(
            {
                "input_sha256": input_hash,
                "gt_sha256": gt_hash,
                "train_sample_ids": natural_sorted(
                    train_by_pair_hash[(input_hash, gt_hash)]
                ),
                "val_sample_ids": natural_sorted(
                    val_by_pair_hash[(input_hash, gt_hash)]
                ),
            }
        )
    return matches


def dataset_file_list_fingerprint(scans: Iterable[DirectoryScan]) -> Tuple[str, int]:
    entries: List[Tuple[str, int, str]] = []
    for scan in scans:
        entries.extend(scan.inventory)
    entries.sort(key=lambda item: item[0].encode("utf-8", errors="surrogateescape"))
    digest = hashlib.sha256()
    for relative_path, size_bytes, kind in entries:
        line = json.dumps(
            [relative_path, size_bytes, kind],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8", errors="surrogateescape")
        digest.update(line)
        digest.update(b"\n")
    return digest.hexdigest(), len(entries)


def unsafe_manifest_value(value: str) -> bool:
    return any(character in value for character in ("\t", "\r", "\n"))


def render_manifest(
    sample_ids: Iterable[str], pairs: Mapping[str, Tuple[ImageRecord, ImageRecord]]
) -> bytes:
    rows: List[str] = []
    for sample_id in natural_sorted(sample_ids):
        input_record, gt_record = pairs[sample_id]
        values = (sample_id, input_record.relative_path, gt_record.relative_path)
        if any(unsafe_manifest_value(value) for value in values):
            raise RuntimeError(f"Manifest-unsafe tab or newline for sample {sample_id!r}")
        rows.append("\t".join(values))
    return ("\n".join(rows) + ("\n" if rows else "")).encode("utf-8")


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def add_issue(
    destination: List[Dict[str, Any]], code: str, message: str, **details: Any
) -> None:
    issue: Dict[str, Any] = {"code": code, "message": message}
    issue.update(details)
    destination.append(issue)


def validate_output_paths(
    root: Path, output_json: Path, output_log: Path, manifest_dir: Path
) -> None:
    resolved_outputs = [
        output_json.resolve(strict=False),
        output_log.resolve(strict=False),
        manifest_dir.resolve(strict=False),
    ]
    for output in resolved_outputs:
        if is_within(output, root):
            raise UsageError(
                f"Output path must not be inside the read-only dataset root: {output}"
            )
    if output_json.resolve(strict=False) == output_log.resolve(strict=False):
        raise UsageError("--output-json and --output-log must be different paths")
    if output_json.exists() and output_json.is_dir():
        raise UsageError(f"--output-json points to a directory: {output_json}")
    if output_log.exists() and output_log.is_dir():
        raise UsageError(f"--output-log points to a directory: {output_log}")
    if manifest_dir.exists() and not manifest_dir.is_dir():
        raise UsageError(f"--manifest-dir is not a directory: {manifest_dir}")

    manifest_outputs = {
        (manifest_dir / filename).resolve(strict=False) for filename in MANIFEST_FILENAMES
    }
    if output_json.resolve(strict=False) in manifest_outputs:
        raise UsageError("--output-json collides with a manifest output path")
    if output_log.resolve(strict=False) in manifest_outputs:
        raise UsageError("--output-log collides with a manifest output path")


def parse_nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_positive_integer(value: str) -> int:
    parsed = parse_nonnegative_integer(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit LSUI19 Train/Val paired images and generate deterministic "
            "train/validation/test manifests only when the audit passes."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="LSUI19 root containing Train/input, Train/GT, Val/input, and Val/GT",
    )
    parser.add_argument(
        "--split-seed",
        type=parse_nonnegative_integer,
        default=DEFAULT_SPLIT_SEED,
        help=f"deterministic split seed (default: {DEFAULT_SPLIT_SEED})",
    )
    parser.add_argument(
        "--validation-count",
        type=parse_positive_integer,
        default=DEFAULT_VALIDATION_COUNT,
        help=f"number of Train-pool samples assigned to validation (default: {DEFAULT_VALIDATION_COUNT})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="machine-readable audit result path",
    )
    parser.add_argument(
        "--output-log",
        type=Path,
        required=True,
        help="human-readable audit summary path",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="directory for train.tsv, validation.tsv, test.tsv, and SPLIT_METADATA.json",
    )
    parser.add_argument(
        "--hash-files",
        action="store_true",
        help="hash every image and detect content leakage across Train and Val",
    )
    return parser


def validate_root(root_argument: Path) -> Path:
    root = root_argument.expanduser().resolve(strict=False)
    if not root.exists():
        raise UsageError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise UsageError(f"Dataset root is not a directory: {root}")
    for _, relative in SOURCE_DIRECTORIES:
        path = root / relative
        if not path.exists():
            raise UsageError(f"Required dataset directory does not exist: {path}")
        if not path.is_dir():
            raise UsageError(f"Required dataset path is not a directory: {path}")
    return root


def collect_audit(
    root: Path,
    image_module: Any,
    split_seed: int,
    validation_count: int,
    hash_files: bool,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Tuple[ImageRecord, ImageRecord]],
    Dict[str, Tuple[ImageRecord, ImageRecord]],
]:
    scans: Dict[str, DirectoryScan] = {}
    for label, relative in SOURCE_DIRECTORIES:
        scans[label] = scan_directory(
            label=label,
            directory=root / relative,
            root=root,
            image_module=image_module,
            hash_files=hash_files,
        )

    train_pair_audit, train_pairs = pair_audit(scans["train_input"], scans["train_gt"])
    val_pair_audit, val_pairs = pair_audit(scans["val_input"], scans["val_gt"])
    train_stems = set(scans["train_input"].records_by_stem()) | set(
        scans["train_gt"].records_by_stem()
    )
    val_stems = set(scans["val_input"].records_by_stem()) | set(
        scans["val_gt"].records_by_stem()
    )
    overlapping_stems = natural_sorted(train_stems & val_stems)

    input_hash_matches: List[Dict[str, Any]] = []
    gt_hash_matches: List[Dict[str, Any]] = []
    pair_hash_matches: List[Dict[str, Any]] = []
    if hash_files:
        input_hash_matches = cross_split_hash_matches(
            scans["train_input"].records, scans["val_input"].records
        )
        gt_hash_matches = cross_split_hash_matches(
            scans["train_gt"].records, scans["val_gt"].records
        )
        pair_hash_matches = identical_pair_hash_matches(train_pairs, val_pairs)

    fingerprint, fingerprint_entry_count = dataset_file_list_fingerprint(scans.values())
    blocking_failures: List[Dict[str, Any]] = []
    audit_warnings: List[Dict[str, Any]] = []

    for split_name, pair_result in (
        ("Train", train_pair_audit),
        ("Val", val_pair_audit),
    ):
        if pair_result["input_count"] != pair_result["gt_count"]:
            add_issue(
                blocking_failures,
                "PAIR_COUNT_MISMATCH",
                f"{split_name} input and GT image counts differ.",
                split=split_name,
                input_count=pair_result["input_count"],
                gt_count=pair_result["gt_count"],
            )
        if pair_result["input_missing_gt"]:
            add_issue(
                blocking_failures,
                "INPUT_MISSING_GT",
                f"{split_name} contains input stems without GT stems.",
                split=split_name,
                sample_ids=pair_result["input_missing_gt"],
            )
        if pair_result["gt_missing_input"]:
            add_issue(
                blocking_failures,
                "GT_MISSING_INPUT",
                f"{split_name} contains GT stems without input stems.",
                split=split_name,
                sample_ids=pair_result["gt_missing_input"],
            )
        if pair_result["pair_size_mismatches"]:
            add_issue(
                blocking_failures,
                "PAIR_SIZE_MISMATCH",
                f"{split_name} contains input/GT pairs with different dimensions.",
                split=split_name,
                pairs=pair_result["pair_size_mismatches"],
            )
        if pair_result["pair_rgb_conversion_failures"]:
            add_issue(
                blocking_failures,
                "PAIR_NOT_RGB_CONVERTIBLE",
                f"{split_name} contains pairs that cannot both be converted to RGB.",
                split=split_name,
                pairs=pair_result["pair_rgb_conversion_failures"],
            )

    for label, scan in scans.items():
        duplicates = scan.duplicate_stems()
        if duplicates:
            add_issue(
                blocking_failures,
                "DUPLICATE_STEM",
                f"{scan.relative_directory} contains duplicate filename stems.",
                directory=scan.relative_directory,
                duplicate_stems=duplicates,
            )
        unreadable = [record.to_dict() for record in scan.records if not record.readable]
        if unreadable:
            add_issue(
                blocking_failures,
                "UNREADABLE_IMAGE",
                f"{scan.relative_directory} contains unreadable or corrupt images.",
                directory=scan.relative_directory,
                images=unreadable,
            )
        if scan.empty_files:
            add_issue(
                blocking_failures,
                "EMPTY_FILE",
                f"{scan.relative_directory} contains empty files.",
                directory=scan.relative_directory,
                files=natural_sorted(scan.empty_files),
            )
        if scan.scan_errors:
            add_issue(
                blocking_failures,
                "SCAN_ERROR",
                f"{scan.relative_directory} could not be scanned completely.",
                directory=scan.relative_directory,
                errors=scan.scan_errors,
            )
        if scan.symlinks:
            add_issue(
                blocking_failures,
                "SYMLINK_NOT_AUDITED",
                f"{scan.relative_directory} contains symlinks; targets were not followed.",
                directory=scan.relative_directory,
                paths=natural_sorted(scan.symlinks),
            )
        hash_failures = [record.to_dict() for record in scan.records if record.hash_error]
        if hash_failures:
            add_issue(
                blocking_failures,
                "FILE_HASH_ERROR",
                f"{scan.relative_directory} contains files that could not be hashed.",
                directory=scan.relative_directory,
                images=hash_failures,
            )

        non_rgb = [
            record.relative_path
            for record in scan.records
            if record.readable and record.mode_group != "RGB"
        ]
        if non_rgb:
            add_issue(
                audit_warnings,
                "NON_RGB_SOURCE_MODE",
                (
                    f"{scan.relative_directory} contains non-RGB source modes; these "
                    "are accepted only because RGB conversion was verified."
                ),
                directory=scan.relative_directory,
                count=len(non_rgb),
                files=natural_sorted(non_rgb),
            )
        if scan.non_image_files:
            add_issue(
                audit_warnings,
                "NON_IMAGE_FILES_PRESENT",
                f"{scan.relative_directory} contains non-image files.",
                directory=scan.relative_directory,
                count=len(scan.non_image_files),
            )

    if overlapping_stems:
        add_issue(
            blocking_failures,
            "TRAIN_VAL_STEM_OVERLAP",
            "Train and Val contain overlapping sample stems.",
            sample_ids=overlapping_stems,
        )

    if len(train_pairs) != EXPECTED_TRAIN_POOL_COUNT:
        add_issue(
            blocking_failures,
            "UNEXPECTED_TRAIN_POOL_COUNT",
            "Observed unique Train pairs do not match the fixed LSUI19 study target.",
            expected=EXPECTED_TRAIN_POOL_COUNT,
            observed=len(train_pairs),
        )
    if len(val_pairs) != EXPECTED_TEST_COUNT:
        add_issue(
            blocking_failures,
            "UNEXPECTED_TEST_COUNT",
            "Observed unique Val pairs do not match the fixed LSUI19 study target.",
            expected=EXPECTED_TEST_COUNT,
            observed=len(val_pairs),
        )
    if validation_count >= len(train_pairs):
        add_issue(
            blocking_failures,
            "INVALID_VALIDATION_COUNT_FOR_POOL",
            "Validation count must be smaller than the audited Train pool.",
            validation_count=validation_count,
            train_pair_count=len(train_pairs),
        )

    unsafe_values: List[Dict[str, str]] = []
    for split_name, pairs in (("Train", train_pairs), ("Val", val_pairs)):
        for sample_id, (input_record, gt_record) in pairs.items():
            for field_name, value in (
                ("sample_id", sample_id),
                ("input_relative_path", input_record.relative_path),
                ("gt_relative_path", gt_record.relative_path),
            ):
                if unsafe_manifest_value(value):
                    unsafe_values.append(
                        {
                            "split": split_name,
                            "sample_id": sample_id,
                            "field": field_name,
                            "value_repr": repr(value),
                        }
                    )
    if unsafe_values:
        add_issue(
            blocking_failures,
            "MANIFEST_UNSAFE_FILENAME",
            "Tabs or newlines in identifiers/paths cannot be represented safely in TSV.",
            values=unsafe_values,
        )

    if hash_files and pair_hash_matches:
        add_issue(
            blocking_failures,
            "CROSS_SPLIT_IDENTICAL_PAIR_CONTENT",
            "Train and Val contain fully identical input/GT pair content.",
            matches=pair_hash_matches,
        )
    if hash_files and input_hash_matches:
        add_issue(
            audit_warnings,
            "CROSS_SPLIT_IDENTICAL_INPUT_CONTENT",
            "Train and Val contain identical input-file content.",
            match_group_count=len(input_hash_matches),
        )
    if hash_files and gt_hash_matches:
        add_issue(
            audit_warnings,
            "CROSS_SPLIT_IDENTICAL_GT_CONTENT",
            "Train and Val contain identical GT-file content.",
            match_group_count=len(gt_hash_matches),
        )
    if not hash_files:
        add_issue(
            audit_warnings,
            "CONTENT_HASH_AUDIT_DISABLED",
            (
                "--hash-files was not enabled; content-based cross-split leakage was "
                "not checked. Stem overlap was still checked."
            ),
        )

    status = "PASS" if not blocking_failures else "FAIL"
    result: Dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_identifier": DATASET_IDENTIFIER,
        "observed_root_basename": root.name,
        "dataset_root": str(root),
        "generated_at_utc": utc_timestamp(),
        "audit_status": status,
        "hash_files_enabled": hash_files,
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "sample_id_semantics": "Exact filename stem; stem comparison is case-sensitive.",
        "non_rgb_policy": {
            "source_modes_recorded": ["RGB", "L", "RGBA", "OTHER"],
            "default_failure_condition": (
                "A pair fails only if either image cannot be fully decoded and converted "
                "to RGB; convertible L, RGBA, and other modes are recorded as warnings."
            ),
        },
        "expected_counts": {
            "train_pool": EXPECTED_TRAIN_POOL_COUNT,
            "held_out_test": EXPECTED_TEST_COUNT,
            "validation": validation_count,
            "derived_train": EXPECTED_TRAIN_POOL_COUNT - validation_count,
        },
        "split_request": {
            "split_seed": split_seed,
            "validation_count": validation_count,
        },
        "directories": {label: scans[label].to_dict() for label, _ in SOURCE_DIRECTORIES},
        "pairing": {"Train": train_pair_audit, "Val": val_pair_audit},
        "cross_split": {
            "overlapping_stems": overlapping_stems,
            "identical_input_content": input_hash_matches if hash_files else None,
            "identical_gt_content": gt_hash_matches if hash_files else None,
            "identical_pair_content": pair_hash_matches if hash_files else None,
        },
        "dataset_file_list_fingerprint": {
            "algorithm": (
                "SHA256 over UTF-8 JSON lines [root-relative-path, byte-size, kind], "
                "sorted by root-relative path bytes"
            ),
            "sha256": fingerprint,
            "entry_count": fingerprint_entry_count,
        },
        "blocking_failures": blocking_failures,
        "warnings": audit_warnings,
        "manifest_generation": {
            "generated": False,
            "reason": (
                "Pending manifest generation after PASS."
                if status == "PASS"
                else "Audit failed; formal split manifests were not generated."
            ),
        },
    }
    return result, train_pairs, val_pairs


def build_split_outputs(
    audit: Dict[str, Any],
    train_pairs: Mapping[str, Tuple[ImageRecord, ImageRecord]],
    val_pairs: Mapping[str, Tuple[ImageRecord, ImageRecord]],
    split_seed: int,
    validation_count: int,
) -> Tuple[Dict[str, bytes], Dict[str, Any]]:
    ranking: List[Tuple[str, str]] = []
    for sample_id in train_pairs:
        material = f"{DATASET_IDENTIFIER}|{split_seed}|{sample_id}".encode("utf-8")
        ranking.append((hashlib.sha256(material).hexdigest(), sample_id))
    ranking.sort(key=lambda item: (item[0], item[1].encode("utf-8")))
    validation_ids = {sample_id for _, sample_id in ranking[:validation_count]}
    train_ids = set(train_pairs) - validation_ids
    test_ids = set(val_pairs)

    manifests = {
        "train.tsv": render_manifest(train_ids, train_pairs),
        "validation.tsv": render_manifest(validation_ids, train_pairs),
        "test.tsv": render_manifest(test_ids, val_pairs),
    }
    manifest_hashes = {
        filename: sha256_bytes(content) for filename, content in manifests.items()
    }
    metadata: Dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_identifier": DATASET_IDENTIFIER,
        "observed_root_basename": audit["observed_root_basename"],
        "generation_timestamp_utc": audit["generated_at_utc"],
        "split_seed": split_seed,
        "validation_count": validation_count,
        "exact_split_algorithm": {
            "ranking_expression": (
                f'sha256("{DATASET_IDENTIFIER}|{split_seed}|<sample_id>")'
            ),
            "ranking_order": "ascending hexadecimal digest; sample_id byte order breaks impossible digest ties",
            "validation_selection": f"first {validation_count} ranked Train-pool sample_ids",
            "train_selection": "all remaining Train-pool sample_ids",
            "test_selection": "all original Val sample_ids",
            "manifest_row_order": "natural ascending sample_id order; independent of ranking order",
        },
        "source_directory_semantics": {
            "Train/input": "degraded inputs for the Train pool",
            "Train/GT": "ground truth paired to Train/input by exact filename stem",
            "Val/input": "degraded inputs held out wholly as test",
            "Val/GT": "ground truth paired to Val/input by exact filename stem",
        },
        "counts": {
            "train": len(train_ids),
            "validation": len(validation_ids),
            "test": len(test_ids),
        },
        "manifest_sha256": manifest_hashes,
        "dataset_file_list_fingerprint": audit["dataset_file_list_fingerprint"],
        "hash_files_enabled": audit["hash_files_enabled"],
        "audit_status": audit["audit_status"],
    }
    metadata_content = json_bytes(metadata)
    outputs = dict(manifests)
    outputs["SPLIT_METADATA.json"] = metadata_content
    return outputs, metadata


def build_log(audit: Mapping[str, Any]) -> str:
    directories = audit["directories"]
    pairing = audit["pairing"]
    cross_split = audit["cross_split"]
    lines = [
        "LSUI19 paired dataset audit",
        f"Status: {audit['audit_status']}",
        f"Observed root basename: {audit['observed_root_basename']}",
        f"Hash files enabled: {audit['hash_files_enabled']}",
        "Image counts:",
    ]
    for label, _ in SOURCE_DIRECTORIES:
        directory = directories[label]
        lines.append(f"  {directory['directory']}: {directory['image_count']}")
    for split_name in ("Train", "Val"):
        pair = pairing[split_name]
        lines.append(
            f"{split_name} pairs: {pair['paired_unique_stem_count']} unique; "
            f"missing GT={len(pair['input_missing_gt'])}; "
            f"missing input={len(pair['gt_missing_input'])}; "
            f"size mismatches={len(pair['pair_size_mismatches'])}; "
            f"RGB-conversion failures={len(pair['pair_rgb_conversion_failures'])}"
        )
    lines.append(f"Train/Val overlapping stems: {len(cross_split['overlapping_stems'])}")
    if audit["hash_files_enabled"]:
        lines.append(
            "Cross-split identical content groups: "
            f"input={len(cross_split['identical_input_content'])}; "
            f"GT={len(cross_split['identical_gt_content'])}; "
            f"full pairs={len(cross_split['identical_pair_content'])}"
        )
    lines.append(f"Blocking failures: {len(audit['blocking_failures'])}")
    for issue in audit["blocking_failures"]:
        lines.append(f"  - {issue['code']}: {issue['message']}")
    lines.append(f"Warnings: {len(audit['warnings'])}")
    for issue in audit["warnings"]:
        lines.append(f"  - {issue['code']}: {issue['message']}")
    manifest = audit["manifest_generation"]
    lines.append(f"Formal manifests generated: {manifest['generated']}")
    lines.append(f"Manifest note: {manifest['reason']}")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    if args.split_seed != DEFAULT_SPLIT_SEED:
        raise UsageError(
            f"This study fixes --split-seed at {DEFAULT_SPLIT_SEED}; "
            f"received {args.split_seed}."
        )
    if args.validation_count != DEFAULT_VALIDATION_COUNT:
        raise UsageError(
            f"This study fixes --validation-count at {DEFAULT_VALIDATION_COUNT}; "
            f"received {args.validation_count}."
        )
    root = validate_root(args.root)
    output_json = args.output_json.expanduser().resolve(strict=False)
    output_log = args.output_log.expanduser().resolve(strict=False)
    manifest_dir = args.manifest_dir.expanduser().resolve(strict=False)
    validate_output_paths(root, output_json, output_log, manifest_dir)
    image_module = load_pillow()

    audit, train_pairs, val_pairs = collect_audit(
        root=root,
        image_module=image_module,
        split_seed=args.split_seed,
        validation_count=args.validation_count,
        hash_files=args.hash_files,
    )

    if audit["audit_status"] == "PASS":
        split_outputs, metadata = build_split_outputs(
            audit=audit,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            split_seed=args.split_seed,
            validation_count=args.validation_count,
        )
        for filename in MANIFEST_FILENAMES:
            atomic_write_bytes(manifest_dir / filename, split_outputs[filename])
        audit["manifest_generation"] = {
            "generated": True,
            "reason": "Audit passed and all formal split manifests were written.",
            "manifest_directory": str(manifest_dir),
            "files": list(MANIFEST_FILENAMES),
            "counts": metadata["counts"],
            "manifest_sha256": metadata["manifest_sha256"],
        }

    log_content = build_log(audit)
    atomic_write_bytes(output_json, json_bytes(audit))
    atomic_write_bytes(output_log, log_content.encode("utf-8"))
    sys.stdout.write(log_content)
    return 0 if audit["audit_status"] == "PASS" else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except UsageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: unexpected failure: {safe_error(exc)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
