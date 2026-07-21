# LSUI19 Dataset Audit

## 1. Audit Status and Evidence Scope

- Status: `PASS`
- Blocking failures: `0`
- Warnings: `0`
- Runtime audit timestamp: `2026-07-20T15:44:45Z`
- Runtime evidence:
  - `reports/runtime/lsui19_dataset_audit.json`
  - `reports/runtime/lsui19_dataset_audit.log`
  - `splits/lsui19/SPLIT_METADATA.json`
- Result-artifact Git commit: `b0bd2e219a34c634408350e25ed3afe72330a119`

This document archives the existing cloud audit outputs. The dataset audit was
not rerun while preparing this document.

## 2. Dataset Roots

- Original data root: `/root/autodl-tmp/pro/publicdata/LSUI`
- Formal physical split root:
  `/root/autodl-tmp/pro/publicdata/LSUI19_dup_train`

The original-root provenance and retain-all-samples policy are defined by
Decision D-0014. The formal physical root is the `dataset_root` recorded by the
runtime audit JSON.

## 3. Physical Split and Semantics

| Physical directory | Observed images | Semantic role |
| --- | ---: | --- |
| `Train/input` | 3851 | Degraded inputs in the development pool |
| `Train/GT` | 3851 | Ground truth paired to the development inputs |
| `Val/input` | 428 | Degraded inputs in the held-out final test set |
| `Val/GT` | 428 | Ground truth paired to the held-out final test inputs |

The exact filename stem is the pair key. Both physical partitions have complete
one-to-one stem pairing, no duplicate stems, no unreadable pairs, no RGB
conversion failures, and no input/GT dimension mismatches.

Formal semantics are frozen as follows:

- Physical `Train` is the development pool.
- Physical `Val` is the final test set.
- Formal train: `3466` samples.
- Formal validation: `385` samples.
- Formal test: `428` samples.

## 4. Duplicate-Aware Policy

- No original sample is deleted.
- Decoded-RGB input duplicate groups in physical Train: `36`.
- Decoded-RGB GT duplicate groups in physical Train: `46`.
- Samples belonging to at least one duplicate input or duplicate GT group are
  combined by `sample_id`, with each sample recorded once.
- Forced formal train samples: `112`.
- All 112 forced samples occur in formal train and none occur in formal
  validation or formal test.
- Validation is selected only from the `3739` non-duplicated development
  samples.
- Physical Val contains `0` decoded-RGB input duplicate groups and `0`
  decoded-RGB GT duplicate groups.

The forced-sample manifest is
`splits/lsui19/forced_formal_train.tsv`.

## 5. Fixed Split Parameters and Algorithm

- Split seed: `3407`.
- Validation count: `385`.
- Validation candidates: all physical Train sample IDs minus all forced formal
  train sample IDs.
- Ranking expression:
  `sha256("LSUI19|3407|<sample_id>")`.
- Ranking order: ascending hexadecimal digest, with sample-ID byte order used
  only to break a digest tie.
- Validation selection: the first 385 ranked candidate sample IDs.
- Formal train: all remaining physical Train sample IDs.
- Formal test: every physical Val sample ID.
- Manifest row order: natural ascending `sample_id` order, independent of
  ranking order.

`splits/lsui19/SPLIT_METADATA.json` is the authoritative record of the stable
sorting and selection algorithm.

## 6. Integrity and Leakage Results

### Physical Train versus Val

| Check | Observed |
| --- | ---: |
| Overlapping `sample_id` stems | 0 |
| Decoded-RGB input hash overlap groups | 0 |
| Decoded-RGB GT hash overlap groups | 0 |
| Decoded-RGB full-pair hash overlap groups | 0 |

### Formal split validation

| Split comparison | Input hash overlaps | GT hash overlaps | Sample-ID overlaps |
| --- | ---: | ---: | ---: |
| Train versus validation | 0 | 0 | 0 |
| Train versus test | 0 | 0 | 0 |
| Validation versus test | 0 | 0 | 0 |

- Unique manifest coverage: `4279` sample IDs.
- Total rows across train, validation, and test manifests: `4279`.
- Missing sample IDs: `0`.
- Unexpected sample IDs: `0`.
- Each of the 4279 sample IDs appears in exactly one primary manifest.
- All manifest image paths are relative to the formal physical split root; no
  manifest contains an absolute dataset path.

## 7. Artifact Fingerprints

Dataset file-list fingerprint:

- Algorithm: SHA256 over UTF-8 JSON lines containing
  `[root-relative-path, byte-size, kind]`, sorted by root-relative path bytes.
- Entry count: `8558`.
- SHA256:
  `33e9c4a8e759438edbda43353663810723c4e8af3c796dc6c682904c75c8ba55`.

Manifest SHA256 values:

| Manifest | Rows | SHA256 |
| --- | ---: | --- |
| `train.tsv` | 3466 | `5cf9be63b7ed565ad3190936c61efe56c7b27c1e0cb7d8b0c9266ef62f87c6ab` |
| `validation.tsv` | 385 | `e81c35ae694ce9c0e2ba656ad5dece093ae7804d4ffd711eeb357103686f9c18` |
| `test.tsv` | 428 | `cee6a22aeb2903f1cd053f641eab3aa1733f55a394682e257cb4ab4b27b0373c` |
| `forced_formal_train.tsv` | 112 | `ae2430f6be1a5ce79ba9a4e95da04ca416d23e70fb75a739bf31b3e06ad7d3d0` |

## 8. Final-Test Isolation Rule

The 428-sample physical Val/formal test set is prohibited from use for:

- checkpoint selection;
- hyperparameter selection;
- router training;
- model selection;
- any feedback during training.

Test inputs and GT may be used only for final, protocol-authorized evaluation
and analysis.

## 9. Final Audit Decision

`PASS`

The duplicate-aware LSUI formal train/validation/test split has been generated
and passed integrity, coverage, forced-assignment, and cross-split decoded-RGB
content-overlap checks. This result does not establish any restoration,
training, metric, operator-order, or paper-performance conclusion.
