# Phase B2a Report — LSUI Dataset and Split Audit

## Phase Objective

Archive the cloud execution results of the duplicate-aware LSUI19 paired-data
audit and freeze reproducible formal train, validation, test, and forced-formal-
train manifests. This phase does not implement or validate a data-loading or
training framework.

## Git Baseline

- Branch: `main`.
- Starting commit for this result-archival task:
  `b0bd2e219a34c634408350e25ed3afe72330a119`.
- Runtime result-artifact commit:
  `b0bd2e219a34c634408350e25ed3afe72330a119`
  (`data: record duplicate-aware LSUI split`).
- Historical Phase B2a start commit in `ORDER_STUDY_STATE.yaml`: not recorded
  (`phase_start_commit: null`).
- Ending commit for the two archival documents: `pending`.

The worktree was clean at the beginning of this archive-only task.

## Changed Files

Files created by this result-archival task:

- `DATASET_AUDIT.md`
- `reports/PHASE_B2A_REPORT.md`

Existing Python code, runtime audit evidence, manifests, governance files, and
the upstream NAFNet repository were not modified.

## Evidence Reviewed

- `ORDER_STUDY_PROTOCOL.md`
- `ORDER_STUDY_DECISIONS.md`
- `ORDER_STUDY_STATE.yaml`
- `tools/audit_lsui19.py`
- `reports/runtime/lsui19_dataset_audit.json`
- `reports/runtime/lsui19_dataset_audit.log`
- `splits/lsui19/train.tsv`
- `splits/lsui19/validation.tsv`
- `splits/lsui19/test.tsv`
- `splits/lsui19/forced_formal_train.tsv`
- `splits/lsui19/SPLIT_METADATA.json`

## Commands

Only read-only archival and consistency checks were executed. The data audit
tool was not rerun.

```text
cat ORDER_STUDY_PROTOCOL.md
cat ORDER_STUDY_DECISIONS.md
cat ORDER_STUDY_STATE.yaml
cat tools/audit_lsui19.py
cat reports/runtime/lsui19_dataset_audit.log
cat splits/lsui19/SPLIT_METADATA.json
git branch --show-current
git rev-parse HEAD
git log -5 --oneline --decorate
git status --short --untracked-files=all
git show --stat --oneline --summary HEAD
python3 - <<'PY'
# Read-only parsing of the existing audit JSON and four TSV manifests:
# validate row structure, relative paths, counts, ID uniqueness/disjointness,
# forced assignment, complete coverage, and SHA256 agreement with metadata.
PY
```

No command accessed the cloud dataset root, decoded an image, regenerated a
hash, or wrote a manifest during this archival task. The original cloud audit
invocation command is not present in the supplied JSON or log and is therefore
not inferred here.

## Actual Outputs

Runtime audit summary:

- Generation timestamp: `2026-07-20T15:44:45Z`.
- Formal physical root:
  `/root/autodl-tmp/pro/publicdata/LSUI19_dup_train`.
- Audit status: `PASS`.
- Blocking failures: `0`.
- Warnings: `0`.
- Physical Train: `3851` input images, `3851` GT images, `3851` unique pairs.
- Physical Val: `428` input images, `428` GT images, `428` unique pairs.
- Train/Val stem overlap: `0`.
- Physical Train/Val decoded-RGB overlaps: input `0`, GT `0`, full pair `0`.
- Train duplicate groups: input `36`, GT `46`.
- Forced formal train samples: `112`.
- Formal split: train `3466`, validation `385`, test `428`.
- Formal cross-split decoded-RGB overlaps: input `0`, GT `0` for all three
  pairwise split comparisons.
- Unique manifest coverage: `4279`.
- Total primary-manifest rows: `4279`.
- Formal split validation status: `PASS`.

Read-only archival verification output:

```text
ARCHIVE_ARTIFACT_VERIFICATION=PASS
counts=train:3466 validation:385 test:428 forced:112 total_unique:4279
sample_id_overlap=train_validation:0 train_test:0 validation_test:0
forced_assignment=missing_train:0 in_validation:0 in_test:0
relative_paths_only=true malformed_rows=0 duplicate_ids_within_manifests=0
manifest_sha256_match_metadata=true
audit_metadata_status_match=PASS
dataset_file_list_fingerprint_match=true
```

## Acceptance Criteria

| Criterion | Actual result | Status |
| --- | --- | --- |
| Physical Train input/GT counts | 3851 / 3851 | PASS |
| Physical Val input/GT counts | 428 / 428 | PASS |
| Complete exact-stem pairing | 3851 Train pairs; 428 Val pairs; no missing pair | PASS |
| Readable and RGB-convertible pairs | 0 failures | PASS |
| Pair dimensions match | 0 mismatches | PASS |
| Physical Train/Val stem overlap | 0 | PASS |
| Physical Train/Val decoded-RGB input, GT, and pair overlap | 0 / 0 / 0 | PASS |
| Duplicate samples forced into formal train | 112 of 112; none in validation/test | PASS |
| Formal split counts | 3466 / 385 / 428 | PASS |
| Primary-manifest sample-ID disjointness | All pairwise overlaps are 0 | PASS |
| Formal cross-split decoded-RGB input and GT overlap | All pairwise overlaps are 0 | PASS |
| Complete manifest coverage | 4279 unique IDs and 4279 total rows | PASS |
| Relative manifest paths | No absolute paths | PASS |
| Manifest SHA256 versus metadata | All four match | PASS |

## Assumptions and Evidence Boundaries

- The original dataset root
  `/root/autodl-tmp/pro/publicdata/LSUI` and the retain-all-samples policy are
  human-approved in Decision D-0014.
- The formal physical root, counts, decoded-RGB hashes, duplicate groups, and
  audit outcome are taken from the immutable runtime JSON/log and metadata.
- This archival step independently checked only the recorded artifacts; it did
  not independently reopen the dataset images or reproduce the cloud audit.
- The test set remains isolated from checkpoint selection, hyperparameter
  selection, router training, model selection, and all training feedback.

## Unresolved Risks

- `ORDER_STUDY_STATE.yaml` still records `git.phase_start_commit: null`; the
  historical Phase B2a starting commit is therefore not established by the
  state file.
- The supplied runtime JSON and log do not record the exact cloud command,
  Python/Pillow versions, host environment, or audit process exit code. These
  values are not inferred.
- Input range, normalization, crop/augmentation policy, metric definitions,
  loss, checkpoint rule, and the Dataset/DataLoader contract remain unresolved
  for future explicitly authorized phases.
- Human review and a Git commit containing these archival documents remain
  required before any phase advancement.

## Supported Conclusions

“重复感知的LSUI正式train/validation/test划分已经生成并通过完整性、覆盖率和跨split内容重叠检查。”

## Unsupported Conclusions

- 尚未实现Dataset或DataLoader；
- 尚未验证训练链路；
- 尚未验证NAFNet在LSUI上的收敛；
- 尚无PSNR或SSIM结果；
- 尚未实现颜色算子；
- 尚未实现散射算子；
- 尚未验证顺序假设；
- 尚无论文实验结论。

## Next Phase Authorization

`false`

No next phase is authorized. This report does not modify
`ORDER_STUDY_STATE.yaml` and does not advance the project automatically.

## Final Phase Status

`PASS`
