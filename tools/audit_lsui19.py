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
EXPECTED_FORMAL_TRAIN_COUNT = 3466
EXPECTED_TOTAL_SAMPLE_COUNT = 4279
DEFAULT_SPLIT_SEED = 3407
DEFAULT_VALIDATION_COUNT = 385
DECODED_RGB_HASH_SCHEMA = "LSUI19_DECODED_RGB_V1"
PAIR_HASH_SCHEMA = "LSUI19_DECODED_RGB_PAIR_V1"
MANIFEST_FILENAMES = (
    "train.tsv",
    "validation.tsv",
    "test.tsv",
    "forced_formal_train.tsv",
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


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def decoded_rgb_sha256(rgb_image: Any) -> str:
    """Hash decoded RGB pixels with explicit dimensions and schema framing."""

    width, height = rgb_image.size
    digest = hashlib.sha256()
    digest.update(DECODED_RGB_HASH_SCHEMA.encode("ascii"))
    digest.update(b"\0")
    digest.update(width.to_bytes(8, byteorder="big", signed=False))
    digest.update(height.to_bytes(8, byteorder="big", signed=False))
    digest.update(rgb_image.tobytes())
    return digest.hexdigest()


def decoded_rgb_pair_sha256(
    input_hash: str,
    gt_hash: str,
) -> str:
    """Combine canonical decoded-RGB input and GT hashes into one pair hash."""

    digest = hashlib.sha256()
    digest.update(PAIR_HASH_SCHEMA.encode("ascii"))
    digest.update(b"\0")
    digest.update(bytes.fromhex(input_hash))
    digest.update(bytes.fromhex(gt_hash))
    return digest.hexdigest()


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
    decoded_rgb_sha256: Optional[str] = None
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
        if self.decoded_rgb_sha256 is not None:
            result["decoded_rgb_sha256"] = self.decoded_rgb_sha256
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


def inspect_image(
    record: ImageRecord,
    image_module: Any,
    hash_decoded_rgb: bool,
) -> None:
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
                        if hash_decoded_rgb:
                            try:
                                record.decoded_rgb_sha256 = decoded_rgb_sha256(converted)
                            except Exception as exc:
                                record.hash_error = safe_error(exc)
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
            inspect_image(record, image_module, hash_decoded_rgb=hash_files)
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


def records_by_decoded_rgb_hash(
    records: Iterable[ImageRecord],
) -> Dict[str, List[ImageRecord]]:
    grouped: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.decoded_rgb_sha256 is not None:
            grouped[record.decoded_rgb_sha256].append(record)
    return dict(grouped)


def duplicate_decoded_rgb_groups(
    records: Iterable[ImageRecord],
) -> List[Dict[str, Any]]:
    """Describe every within-collection decoded-RGB content group of size > 1."""

    groups: List[Dict[str, Any]] = []
    for digest, grouped_records in sorted(records_by_decoded_rgb_hash(records).items()):
        if len(grouped_records) <= 1:
            continue
        samples = sorted(
            (
                {
                    "sample_id": record.sample_id,
                    "relative_path": record.relative_path,
                }
                for record in grouped_records
            ),
            key=lambda item: natural_sort_key(str(item["sample_id"])),
        )
        groups.append(
            {
                "decoded_rgb_sha256": digest,
                "sample_count": len(samples),
                "samples": samples,
            }
        )
    return groups


def sample_ids_in_duplicate_groups(groups: Iterable[Mapping[str, Any]]) -> set:
    sample_ids = set()
    for group in groups:
        for sample in group["samples"]:
            sample_ids.add(str(sample["sample_id"]))
    return sample_ids


def cross_split_hash_matches(
    left_records: Iterable[ImageRecord], right_records: Iterable[ImageRecord]
) -> List[Dict[str, Any]]:
    left_by_hash = records_by_decoded_rgb_hash(left_records)
    right_by_hash = records_by_decoded_rgb_hash(right_records)

    matches: List[Dict[str, Any]] = []
    for digest in sorted(set(left_by_hash) & set(right_by_hash)):
        matches.append(
            {
                "decoded_rgb_sha256": digest,
                "left": sorted(
                    (
                        {
                            "sample_id": record.sample_id,
                            "relative_path": record.relative_path,
                        }
                        for record in left_by_hash[digest]
                    ),
                    key=lambda item: natural_sort_key(str(item["relative_path"])),
                ),
                "right": sorted(
                    (
                        {
                            "sample_id": record.sample_id,
                            "relative_path": record.relative_path,
                        }
                        for record in right_by_hash[digest]
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
    train_by_pair_hash: Dict[str, List[str]] = defaultdict(list)
    val_by_pair_hash: Dict[str, List[str]] = defaultdict(list)
    for sample_id, (input_record, gt_record) in train_pairs.items():
        if (
            input_record.decoded_rgb_sha256 is not None
            and gt_record.decoded_rgb_sha256 is not None
        ):
            pair_hash = decoded_rgb_pair_sha256(
                input_record.decoded_rgb_sha256,
                gt_record.decoded_rgb_sha256,
            )
            train_by_pair_hash[pair_hash].append(sample_id)
    for sample_id, (input_record, gt_record) in val_pairs.items():
        if (
            input_record.decoded_rgb_sha256 is not None
            and gt_record.decoded_rgb_sha256 is not None
        ):
            pair_hash = decoded_rgb_pair_sha256(
                input_record.decoded_rgb_sha256,
                gt_record.decoded_rgb_sha256,
            )
            val_by_pair_hash[pair_hash].append(sample_id)

    matches: List[Dict[str, Any]] = []
    for pair_hash in sorted(set(train_by_pair_hash) & set(val_by_pair_hash)):
        matches.append(
            {
                "decoded_rgb_pair_sha256": pair_hash,
                "train_sample_ids": natural_sorted(
                    train_by_pair_hash[pair_hash]
                ),
                "val_sample_ids": natural_sorted(
                    val_by_pair_hash[pair_hash]
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


def select_component_records(
    sample_ids: Iterable[str],
    pairs: Mapping[str, Tuple[ImageRecord, ImageRecord]],
    component_index: int,
) -> List[ImageRecord]:
    return [pairs[sample_id][component_index] for sample_id in sample_ids]


def plan_duplicate_aware_split(
    train_pairs: Mapping[str, Tuple[ImageRecord, ImageRecord]],
    val_pairs: Mapping[str, Tuple[ImageRecord, ImageRecord]],
    forced_formal_train_ids: set,
    split_seed: int,
    validation_count: int,
) -> Dict[str, set]:
    validation_candidates = set(train_pairs) - forced_formal_train_ids
    ranking: List[Tuple[str, str]] = []
    for sample_id in validation_candidates:
        material = f"{DATASET_IDENTIFIER}|{split_seed}|{sample_id}".encode("utf-8")
        ranking.append((hashlib.sha256(material).hexdigest(), sample_id))
    ranking.sort(key=lambda item: (item[0], item[1].encode("utf-8")))
    validation_ids = {sample_id for _, sample_id in ranking[:validation_count]}
    train_ids = set(train_pairs) - validation_ids
    return {
        "train": train_ids,
        "validation": validation_ids,
        "test": set(val_pairs),
        "validation_candidates": validation_candidates,
    }


def validate_formal_split(
    split_ids: Mapping[str, set],
    train_pairs: Mapping[str, Tuple[ImageRecord, ImageRecord]],
    val_pairs: Mapping[str, Tuple[ImageRecord, ImageRecord]],
    forced_formal_train_ids: set,
) -> Dict[str, Any]:
    """Recheck counts, identity coverage, and decoded-RGB leakage before writes."""

    train_ids = split_ids["train"]
    validation_ids = split_ids["validation"]
    test_ids = split_ids["test"]
    split_pair_maps = {
        "train": train_pairs,
        "validation": train_pairs,
        "test": val_pairs,
    }
    named_ids = {
        "train": train_ids,
        "validation": validation_ids,
        "test": test_ids,
    }

    expected_counts = {
        "train": EXPECTED_FORMAL_TRAIN_COUNT,
        "validation": DEFAULT_VALIDATION_COUNT,
        "test": EXPECTED_TEST_COUNT,
    }
    observed_counts = {name: len(ids) for name, ids in named_ids.items()}
    count_checks = {
        name: {
            "expected": expected_counts[name],
            "observed": observed_counts[name],
            "passed": observed_counts[name] == expected_counts[name],
        }
        for name in ("train", "validation", "test")
    }

    identity_overlap: Dict[str, Dict[str, Any]] = {}
    hash_overlap: Dict[str, Dict[str, Any]] = {}
    split_name_pairs = (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    )
    for left_name, right_name in split_name_pairs:
        label = f"{left_name}_vs_{right_name}"
        overlapping_ids = natural_sorted(named_ids[left_name] & named_ids[right_name])
        identity_overlap[label] = {
            "count": len(overlapping_ids),
            "sample_ids": overlapping_ids,
            "passed": not overlapping_ids,
        }

        left_input = select_component_records(
            named_ids[left_name], split_pair_maps[left_name], 0
        )
        right_input = select_component_records(
            named_ids[right_name], split_pair_maps[right_name], 0
        )
        left_gt = select_component_records(
            named_ids[left_name], split_pair_maps[left_name], 1
        )
        right_gt = select_component_records(
            named_ids[right_name], split_pair_maps[right_name], 1
        )
        input_matches = cross_split_hash_matches(left_input, right_input)
        gt_matches = cross_split_hash_matches(left_gt, right_gt)
        hash_overlap[label] = {
            "input_decoded_rgb_hash_overlap_count": len(input_matches),
            "input_decoded_rgb_hash_overlaps": input_matches,
            "gt_decoded_rgb_hash_overlap_count": len(gt_matches),
            "gt_decoded_rgb_hash_overlaps": gt_matches,
            "passed": not input_matches and not gt_matches,
        }

    physical_universe = set(train_pairs) | set(val_pairs)
    manifested_ids = train_ids | validation_ids | test_ids
    total_manifest_rows = len(train_ids) + len(validation_ids) + len(test_ids)
    coverage = {
        "expected_total_sample_count": EXPECTED_TOTAL_SAMPLE_COUNT,
        "physical_unique_sample_count": len(physical_universe),
        "manifest_unique_sample_count": len(manifested_ids),
        "manifest_total_row_count": total_manifest_rows,
        "missing_sample_ids": natural_sorted(physical_universe - manifested_ids),
        "unexpected_sample_ids": natural_sorted(manifested_ids - physical_universe),
    }
    coverage["passed"] = (
        len(physical_universe) == EXPECTED_TOTAL_SAMPLE_COUNT
        and len(manifested_ids) == EXPECTED_TOTAL_SAMPLE_COUNT
        and total_manifest_rows == EXPECTED_TOTAL_SAMPLE_COUNT
        and not coverage["missing_sample_ids"]
        and not coverage["unexpected_sample_ids"]
    )

    forced_missing_from_train = natural_sorted(forced_formal_train_ids - train_ids)
    forced_in_validation = natural_sorted(forced_formal_train_ids & validation_ids)
    forced_in_test = natural_sorted(forced_formal_train_ids & test_ids)
    forced_assignment = {
        "forced_formal_train_count": len(forced_formal_train_ids),
        "missing_from_train": forced_missing_from_train,
        "present_in_validation": forced_in_validation,
        "present_in_test": forced_in_test,
        "passed": (
            not forced_missing_from_train
            and not forced_in_validation
            and not forced_in_test
        ),
    }

    candidate_membership = {
        "validation_candidate_count": len(split_ids["validation_candidates"]),
        "validation_outside_candidates": natural_sorted(
            validation_ids - split_ids["validation_candidates"]
        ),
    }
    candidate_membership["passed"] = not candidate_membership[
        "validation_outside_candidates"
    ]

    all_checks_passed = (
        all(check["passed"] for check in count_checks.values())
        and all(check["passed"] for check in identity_overlap.values())
        and all(check["passed"] for check in hash_overlap.values())
        and bool(coverage["passed"])
        and bool(forced_assignment["passed"])
        and bool(candidate_membership["passed"])
    )
    return {
        "status": "PASS" if all_checks_passed else "FAIL",
        "count_checks": count_checks,
        "sample_id_overlap_checks": identity_overlap,
        "decoded_rgb_hash_overlap_checks": hash_overlap,
        "forced_formal_train_assignment": forced_assignment,
        "validation_candidate_membership": candidate_membership,
        "complete_sample_coverage": coverage,
    }


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
        help=(
            "directory for train.tsv, validation.tsv, test.tsv, "
            "forced_formal_train.tsv, and SPLIT_METADATA.json"
        ),
    )
    parser.add_argument(
        "--hash-files",
        action="store_true",
        help=(
            "required for formal B2a execution; hash canonical decoded-RGB pixels "
            "for duplicate and leakage checks"
        ),
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
    set,
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

    train_duplicate_input_groups = (
        duplicate_decoded_rgb_groups(scans["train_input"].records)
        if hash_files
        else []
    )
    train_duplicate_gt_groups = (
        duplicate_decoded_rgb_groups(scans["train_gt"].records)
        if hash_files
        else []
    )
    val_duplicate_input_groups = (
        duplicate_decoded_rgb_groups(scans["val_input"].records)
        if hash_files
        else []
    )
    val_duplicate_gt_groups = (
        duplicate_decoded_rgb_groups(scans["val_gt"].records)
        if hash_files
        else []
    )
    duplicate_input_sample_ids = sample_ids_in_duplicate_groups(
        train_duplicate_input_groups
    )
    duplicate_gt_sample_ids = sample_ids_in_duplicate_groups(train_duplicate_gt_groups)
    forced_formal_train_ids = duplicate_input_sample_ids | duplicate_gt_sample_ids

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

    expected_directory_counts = {
        "train_input": EXPECTED_TRAIN_POOL_COUNT,
        "train_gt": EXPECTED_TRAIN_POOL_COUNT,
        "val_input": EXPECTED_TEST_COUNT,
        "val_gt": EXPECTED_TEST_COUNT,
    }
    for label, expected_count in expected_directory_counts.items():
        observed_count = len(scans[label].records)
        if observed_count != expected_count:
            add_issue(
                blocking_failures,
                "UNEXPECTED_PHYSICAL_DIRECTORY_COUNT",
                f"{scans[label].relative_directory} does not match the fixed physical count.",
                directory=scans[label].relative_directory,
                expected=expected_count,
                observed=observed_count,
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

    if hash_files and input_hash_matches:
        add_issue(
            blocking_failures,
            "CROSS_SPLIT_IDENTICAL_INPUT_CONTENT",
            "Physical Train and Val contain overlapping decoded-RGB input content.",
            matches=input_hash_matches,
        )
    if hash_files and gt_hash_matches:
        add_issue(
            blocking_failures,
            "CROSS_SPLIT_IDENTICAL_GT_CONTENT",
            "Physical Train and Val contain overlapping decoded-RGB GT content.",
            matches=gt_hash_matches,
        )
    if hash_files and pair_hash_matches:
        add_issue(
            blocking_failures,
            "CROSS_SPLIT_IDENTICAL_PAIR_CONTENT",
            "Physical Train and Val contain overlapping decoded-RGB full-pair content.",
            matches=pair_hash_matches,
        )
    if hash_files and val_duplicate_input_groups:
        add_issue(
            blocking_failures,
            "VAL_INTERNAL_DUPLICATE_INPUT_CONTENT",
            (
                "Physical Val contains decoded-RGB input duplicate groups, so not all "
                "duplicated input samples were forced into physical Train."
            ),
            groups=val_duplicate_input_groups,
        )
    if hash_files and val_duplicate_gt_groups:
        add_issue(
            blocking_failures,
            "VAL_INTERNAL_DUPLICATE_GT_CONTENT",
            (
                "Physical Val contains decoded-RGB GT duplicate groups, so not all "
                "duplicated GT samples were forced into physical Train."
            ),
            groups=val_duplicate_gt_groups,
        )
    if not hash_files:
        add_issue(
            blocking_failures,
            "DECODED_RGB_HASH_AUDIT_REQUIRED",
            (
                "--hash-files is required for the formal B2a audit because decoded-RGB "
                "duplicate and leakage checks are mandatory."
            ),
        )

    validation_candidate_count = len(set(train_pairs) - forced_formal_train_ids)
    if validation_candidate_count < validation_count:
        add_issue(
            blocking_failures,
            "INSUFFICIENT_NON_DUPLICATE_VALIDATION_CANDIDATES",
            "Too few non-duplicated Train samples remain for formal validation.",
            required=validation_count,
            observed=validation_candidate_count,
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
        "content_hash_semantics": {
            "schema": DECODED_RGB_HASH_SCHEMA,
            "algorithm": (
                "SHA256(schema + NUL + width:uint64be + height:uint64be + "
                "Pillow-decoded RGB pixel bytes in row-major order)"
            ),
            "pair_schema": PAIR_HASH_SCHEMA,
            "pair_algorithm": (
                "SHA256(pair-schema + NUL + binary decoded-RGB input digest + "
                "binary decoded-RGB GT digest)"
            ),
        },
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
            "decoded_rgb_input_hash_overlaps": (
                input_hash_matches if hash_files else None
            ),
            "decoded_rgb_gt_hash_overlaps": gt_hash_matches if hash_files else None,
            "decoded_rgb_full_pair_hash_overlaps": (
                pair_hash_matches if hash_files else None
            ),
        },
        "train_internal_duplicates": {
            "decoded_rgb_input_duplicate_group_count": len(
                train_duplicate_input_groups
            ),
            "decoded_rgb_gt_duplicate_group_count": len(train_duplicate_gt_groups),
            "decoded_rgb_input_duplicate_groups": train_duplicate_input_groups,
            "decoded_rgb_gt_duplicate_groups": train_duplicate_gt_groups,
            "duplicate_input_sample_ids": natural_sorted(duplicate_input_sample_ids),
            "duplicate_gt_sample_ids": natural_sorted(duplicate_gt_sample_ids),
            "forced_formal_train_sample_count": len(forced_formal_train_ids),
            "forced_formal_train_sample_ids": natural_sorted(forced_formal_train_ids),
            "validation_candidate_count": validation_candidate_count,
        },
        "val_internal_duplicates": {
            "decoded_rgb_input_duplicate_group_count": len(
                val_duplicate_input_groups
            ),
            "decoded_rgb_gt_duplicate_group_count": len(val_duplicate_gt_groups),
            "decoded_rgb_input_duplicate_groups": val_duplicate_input_groups,
            "decoded_rgb_gt_duplicate_groups": val_duplicate_gt_groups,
            "required_group_count": 0,
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
        "formal_split_validation": {
            "status": "NOT_RUN",
            "reason": "Physical audit must pass before formal split planning.",
        },
    }
    return result, train_pairs, val_pairs, forced_formal_train_ids


def build_split_outputs(
    audit: Dict[str, Any],
    train_pairs: Mapping[str, Tuple[ImageRecord, ImageRecord]],
    val_pairs: Mapping[str, Tuple[ImageRecord, ImageRecord]],
    split_ids: Mapping[str, set],
    forced_formal_train_ids: set,
    formal_split_validation: Mapping[str, Any],
    split_seed: int,
    validation_count: int,
) -> Tuple[Dict[str, bytes], Dict[str, Any]]:
    train_ids = split_ids["train"]
    validation_ids = split_ids["validation"]
    test_ids = split_ids["test"]

    manifests = {
        "train.tsv": render_manifest(train_ids, train_pairs),
        "validation.tsv": render_manifest(validation_ids, train_pairs),
        "test.tsv": render_manifest(test_ids, val_pairs),
        "forced_formal_train.tsv": render_manifest(
            forced_formal_train_ids, train_pairs
        ),
    }
    primary_manifest_hashes = {
        filename: sha256_bytes(manifests[filename])
        for filename in ("train.tsv", "validation.tsv", "test.tsv")
    }
    forced_manifest_hash = sha256_bytes(manifests["forced_formal_train.tsv"])
    duplicate_analysis = audit["train_internal_duplicates"]
    metadata: Dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_identifier": DATASET_IDENTIFIER,
        "observed_root_basename": audit["observed_root_basename"],
        "generation_timestamp_utc": audit["generated_at_utc"],
        "split_seed": split_seed,
        "validation_count": validation_count,
        "physical_counts": {
            "Train": {
                "input": audit["pairing"]["Train"]["input_count"],
                "GT": audit["pairing"]["Train"]["gt_count"],
                "pairs": audit["pairing"]["Train"]["paired_unique_stem_count"],
            },
            "Val": {
                "input": audit["pairing"]["Val"]["input_count"],
                "GT": audit["pairing"]["Val"]["gt_count"],
                "pairs": audit["pairing"]["Val"]["paired_unique_stem_count"],
            },
        },
        "formal_counts": {
            "train": len(train_ids),
            "validation": len(validation_ids),
            "test": len(test_ids),
        },
        "duplicate_content_summary": {
            "decoded_rgb_input_duplicate_group_count": duplicate_analysis[
                "decoded_rgb_input_duplicate_group_count"
            ],
            "decoded_rgb_gt_duplicate_group_count": duplicate_analysis[
                "decoded_rgb_gt_duplicate_group_count"
            ],
            "forced_formal_train_sample_count": duplicate_analysis[
                "forced_formal_train_sample_count"
            ],
        },
        "content_hash_semantics": audit["content_hash_semantics"],
        "exact_split_algorithm": {
            "ranking_expression": (
                f'sha256("{DATASET_IDENTIFIER}|{split_seed}|<sample_id>")'
            ),
            "ranking_order": "ascending hexadecimal digest; sample_id byte order breaks impossible digest ties",
            "forced_formal_train_definition": (
                "Union of Train sample_ids in any decoded-RGB input duplicate group "
                "or decoded-RGB GT duplicate group"
            ),
            "validation_candidates": (
                "all physical Train sample_ids minus forced_formal_train sample_ids"
            ),
            "validation_selection": (
                f"first {validation_count} ranked validation-candidate sample_ids"
            ),
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
        "manifest_sha256": primary_manifest_hashes,
        "forced_formal_train_manifest_sha256": forced_manifest_hash,
        "split_hash_overlap_validation": formal_split_validation[
            "decoded_rgb_hash_overlap_checks"
        ],
        "formal_split_validation": formal_split_validation,
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
        f"Decoded-RGB hashing enabled: {audit['hash_files_enabled']}",
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
            "Physical Train/Val decoded-RGB hash overlap groups: "
            f"input={len(cross_split['decoded_rgb_input_hash_overlaps'])}; "
            f"GT={len(cross_split['decoded_rgb_gt_hash_overlaps'])}; "
            f"full pairs={len(cross_split['decoded_rgb_full_pair_hash_overlaps'])}"
        )
    duplicates = audit["train_internal_duplicates"]
    lines.append(
        "Train decoded-RGB duplicate groups: "
        f"input={duplicates['decoded_rgb_input_duplicate_group_count']}; "
        f"GT={duplicates['decoded_rgb_gt_duplicate_group_count']}; "
        f"forced formal train samples={duplicates['forced_formal_train_sample_count']}"
    )
    val_duplicates = audit["val_internal_duplicates"]
    lines.append(
        "Val decoded-RGB duplicate groups (required 0): "
        f"input={val_duplicates['decoded_rgb_input_duplicate_group_count']}; "
        f"GT={val_duplicates['decoded_rgb_gt_duplicate_group_count']}"
    )
    formal_validation = audit["formal_split_validation"]
    lines.append(f"Formal split validation: {formal_validation['status']}")
    if formal_validation["status"] in {"PASS", "FAIL"}:
        counts = formal_validation["count_checks"]
        lines.append(
            "Formal counts: "
            f"train={counts['train']['observed']}; "
            f"validation={counts['validation']['observed']}; "
            f"test={counts['test']['observed']}"
        )
        overlap_checks = formal_validation["decoded_rgb_hash_overlap_checks"]
        input_overlap_count = sum(
            check["input_decoded_rgb_hash_overlap_count"]
            for check in overlap_checks.values()
        )
        gt_overlap_count = sum(
            check["gt_decoded_rgb_hash_overlap_count"]
            for check in overlap_checks.values()
        )
        lines.append(
            "Formal cross-split decoded-RGB overlap groups: "
            f"input={input_overlap_count}; GT={gt_overlap_count}"
        )
        coverage = formal_validation["complete_sample_coverage"]
        lines.append(
            "Formal manifest sample coverage: "
            f"unique={coverage['manifest_unique_sample_count']}; "
            f"rows={coverage['manifest_total_row_count']}; "
            f"expected={coverage['expected_total_sample_count']}"
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
    if not args.hash_files:
        raise UsageError(
            "The formal Phase B2a audit requires --hash-files because decoded-RGB "
            "duplicate detection and cross-split leakage checks are mandatory."
        )
    root = validate_root(args.root)
    output_json = args.output_json.expanduser().resolve(strict=False)
    output_log = args.output_log.expanduser().resolve(strict=False)
    manifest_dir = args.manifest_dir.expanduser().resolve(strict=False)
    validate_output_paths(root, output_json, output_log, manifest_dir)
    image_module = load_pillow()

    audit, train_pairs, val_pairs, forced_formal_train_ids = collect_audit(
        root=root,
        image_module=image_module,
        split_seed=args.split_seed,
        validation_count=args.validation_count,
        hash_files=args.hash_files,
    )

    if audit["audit_status"] == "PASS":
        split_ids = plan_duplicate_aware_split(
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            forced_formal_train_ids=forced_formal_train_ids,
            split_seed=args.split_seed,
            validation_count=args.validation_count,
        )
        formal_split_validation = validate_formal_split(
            split_ids=split_ids,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            forced_formal_train_ids=forced_formal_train_ids,
        )
        audit["formal_split_validation"] = formal_split_validation
        if formal_split_validation["status"] != "PASS":
            add_issue(
                audit["blocking_failures"],
                "FORMAL_SPLIT_VALIDATION_FAILED",
                "The duplicate-aware formal split failed final count, coverage, or leakage validation.",
                validation=formal_split_validation,
            )
            audit["audit_status"] = "FAIL"
            audit["manifest_generation"] = {
                "generated": False,
                "reason": (
                    "Final formal split validation failed; no split manifests were generated."
                ),
            }

    if audit["audit_status"] == "PASS":
        split_outputs, metadata = build_split_outputs(
            audit=audit,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            split_ids=split_ids,
            forced_formal_train_ids=forced_formal_train_ids,
            formal_split_validation=formal_split_validation,
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
            "counts": metadata["formal_counts"],
            "manifest_sha256": metadata["manifest_sha256"],
            "forced_formal_train_manifest_sha256": metadata[
                "forced_formal_train_manifest_sha256"
            ],
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
