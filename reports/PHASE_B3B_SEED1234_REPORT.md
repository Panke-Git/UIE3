# Phase B3b-2r1 Formal Seed 1234 Result Archive Report

## 1. Phase Objective and Governance

This report-only phase archives the completed formal NAFNet-small baseline
training and formal-validation result for seed 1234. It does not modify or
rerun training, evaluation, source code, configuration, manifests, governance
files, runtime evidence, checkpoints, or dataset files.

- Protocol version: `1.0.1`
- Archival phase ID: `B3b-2r1`
- Archival phase name: `formal_seed_archive_completion`
- Formal-run phase: `B3b-2`
- Seed: `1234`
- Current phase authorized: `true`
- Next phase authorized: `false`
- Required report: `reports/PHASE_B3B_SEED1234_REPORT.md`
- Final test evaluated: `false`
- Seed 2027 authorized by this report: `false`

The phase-specific prompt narrows the current state authorization to creation
of this report only. No conflict was found among the protocol, accepted
decisions, current state, and the phase-specific archival instructions.

## 2. Commits and Run Identity

- Formal-run Git commit:
  `0a713111f34a3bccacecbadd6131bd641b357c7f`
- Git commit recorded by the runtime environment:
  `0a713111f34a3bccacecbadd6131bd641b357c7f`
- Git commit recorded by `best.pt`:
  `0a713111f34a3bccacecbadd6131bd641b357c7f`
- Git commit recorded by `last.pt`:
  `0a713111f34a3bccacecbadd6131bd641b357c7f`
- Archive-context commit recorded as `git_commit_current` in
  `run_summary.json`:
  `2b22e57960f177201ab3bf17c035f5f139bcba67`
- Repository HEAD at the start of report authoring:
  `7594575e6ae4403ec4acecfcc94e811468f0b042`
- Ending report commit: pending; this report has not been committed
- Run origin: `epoch_0`
- Formal configuration:
  `configs/nafnet_small_lsui_formal_seed1234.yaml`
- Configuration SHA256:
  `5b1dde02deec9984d17760224be975bed3b1a45d25f5b399e85fb27ec0c1e396`
- Formal train manifest: `splits/lsui19/train.tsv`
- Formal validation manifest: `splits/lsui19/validation.tsv`
- Output directory:
  `/root/autodl-tmp/pro/UIE3_runs/b3b/seed_1234`

The runtime log records `RUN_ORIGIN=EPOCH_0`, begins with epoch 0, and contains
one continuous record for every epoch through 199. The archive-context commit
is metadata for the later summary extraction and is not represented as the
formal-run code commit.

## 3. Changed Files

The only file created in this report-only phase is:

- `reports/PHASE_B3B_SEED1234_REPORT.md`

No source, test, configuration, manifest, governance, runtime-evidence,
checkpoint, or dataset file was modified.

## 4. Assumptions and Evidence Boundary

- Runtime facts are taken only from the governance, configuration, runtime log,
  validation CSV, and summary files required by the phase-specific prompt.
- Checkpoint validity is assessed from the recorded SHA256, nonzero size,
  epoch, global step, formal-run commit, and state-presence fields in
  `run_summary.json`. The checkpoint files were not loaded again in this
  report-only phase.
- `run_summary.json` was already present at report-authoring start and is
  treated as read-only evidence.
- The independent best-checkpoint validation CSV was parsed read-only with the
  Python standard library. No model code, checkpoint, or dataset image was
  accessed.
- The exact original cloud shell launch commands are not present in the
  supplied evidence. The logs contain the resolved run identity and
  configuration, but this report does not infer missing literal commands.

## 5. Commands

No training, validation, test, or final-test command was run in this archival
phase. The report was prepared using read-only inspection:

- `sed`, `rg`, `wc`, and `shasum -a 256` for the required governance,
  configuration, log, CSV, JSON, and report evidence;
- `git status`, `git log`, `git branch --show-current`, and
  `git rev-parse HEAD`;
- a Python-standard-library-only read-only parser for epoch continuity,
  finite-value checks, CSV row and unique-ID counts, metric means, and
  cross-file consistency.

The exact formal training and validation launch commands are not reproduced
because they are absent from the supplied evidence.

## 6. Environment and Fixed Formal Configuration

The archived cloud environment records:

- Platform: `Linux-5.15.0-139-generic-x86_64-with-glibc2.35`
- Python: `3.12.13`
- Python executable: `/root/miniconda3/envs/pax/bin/python`
- PyTorch: `2.13.0+cu126`
- CUDA runtime: `12.6`
- cuDNN: `91002`
- GPU: `NVIDIA GeForce RTX 3090`
- Device used by training: `cuda`
- Model parameter count: `2,846,755`
- Run mode: `FORMAL`
- Formal experiment flag: `true`

The resolved configuration records the frozen NAFNet-small model, seed 1234,
formal train and validation manifests, patch size 256, batch size 4, four
workers, horizontal and vertical flips, 90-degree rotation augmentation,
Charbonnier loss with epsilon `0.001`, AdamW with learning rate `0.0002`,
weight decay `0`, AMP enabled, a 200-epoch maximum budget, validation every
epoch, and periodic checkpoint retention every 10 epochs. No test manifest is
present.

The locally computed configuration SHA256 exactly matches the value recorded
in both `environment.txt` and `run_summary.json`.

## 7. Formal Training Completed Epochs 0–199

- Formal run start: `2026-07-23T23:37:54+08:00`
- Formal run end: `2026-07-24T06:32:59+08:00`
- Total training elapsed time: `24,905` seconds (`6 h 55 min 5 s`)
- Training exit code: `0`
- Logged epoch records: `200`
- Completed epoch range: `0` through `199`
- Actual completed epoch count: `200`
- Epoch sequence complete: `true`
- All logged train-loss, validation-PSNR, validation-SSIM,
  best-validation-PSNR, and learning-rate values finite: `true`

The run began at epoch 0 and completed the full frozen 200-epoch budget. The
log records 69 handled AMP overflow events; these did not terminate the run.
As defined by the remediated trainer, checkpoint global steps count successful
optimizer updates rather than attempted minibatches.

## 8. Checkpoints and Selection Rule

Checkpoint selection used the frozen rule:

> highest formal-validation RGB PSNR

| Checkpoint | Epoch | Global step | Size (bytes) | SHA256 |
| --- | ---: | ---: | ---: | --- |
| `best.pt` | 180 | 156,865 | 34,523,299 | `f45d08e0fd3854c9d7c71a635a469af7d32f3207cc558ebbe065e52da2efc667` |
| `last.pt` | 199 | 173,331 | 34,523,299 | `688382edc2b7679e6010452d830a41a7764d58c9c6d3f055a7d83a2b05bd8865` |

For both checkpoints, `run_summary.json` records:

- model state present: `true`;
- optimizer state present: `true`;
- GradScaler state present: `true`;
- the formal-run Git commit:
  `0a713111f34a3bccacecbadd6131bd641b357c7f`.

The training-time selector record for epoch 180 is:

- validation RGB PSNR: `27.334571999388857`;
- validation RGB SSIM: `0.8927692051057692`;
- `is_best: true`;
- global step: `156865`.

`best.pt` and `last.pt` are therefore recorded as nonempty, hashed
checkpoints with complete model, optimizer, and GradScaler state.

## 9. Complete Best-Checkpoint Formal Validation

The archived post-training evaluation used `best.pt` and the formal validation
manifest.

- Validation split: `validation`
- Validation exit code: `0`
- Validation elapsed time: `21` seconds
- Expected formal-validation samples: `385`
- CSV data rows: `385`
- Unique CSV sample IDs: `385`
- CSV columns:
  `sample_id`, `input_relative_path`, `gt_relative_path`, `psnr_rgb`,
  `ssim_rgb`
- Every row contains PSNR and SSIM: `true`
- All 770 per-image PSNR and SSIM values finite: `true`
- Mean PSNR RGB: `27.334599338878284`
- Mean SSIM RGB: `0.8927693091429674`
- Final test evaluated: `false`

The means recomputed directly from the 385 CSV rows exactly match
`run_summary.json`:

- PSNR absolute delta: `0.0`;
- SSIM absolute delta: `0.0`.

The training-time checkpoint-selection record and the independent per-image
CSV mean are numerically consistent within the declared archival tolerance,
but they are not bit-for-bit identical:

- RGB PSNR absolute delta:
  `2.7339489427902208e-05` dB;
- declared PSNR consistency tolerance: `0.0001` dB;
- RGB SSIM absolute delta:
  `1.0403719818974366e-07`.

The PSNR delta is below the declared tolerance, and
`run_summary.json` records `consistent_with_checkpoint_metric=true`. The
evidence does not establish the cause of the small differences, so this report
does not infer one. The selected checkpoint is epoch 180 in both the training
log and checkpoint metadata, and the formal reported metrics above are the
independently archived 385-image CSV means.

This is the formal validation result for seed 1234. It is not a final-test
result and is not the final three-seed baseline summary.

## 10. Actual Outputs

- `train.log` ends with `TRAIN_EXIT_CODE=0`.
- `train_exit_code.txt` records `train_exit_code=0`.
- `run_summary.json` records `FORMAL_SEED_RUN_COMPLETED`.
- `best_validation.log` records `validation_exit_code=0`,
  `num_samples=385`, `split=validation`, and
  `final_test_evaluated=false`.
- `best_validation_metrics.csv` contains one header and 385 data rows.
- `best.pt` and `last.pt` have recorded nonzero sizes, SHA256 values, epochs,
  global steps, formal-run commits, and complete training-state metadata.
- No final-test metric or final-test CSV was produced by the recorded run.

## 11. Acceptance Criteria

| Criterion | Result | Evidence |
| --- | --- | --- |
| `train_exit_code=0` | PASS | `train.log`, `train_exit_code.txt`, and `run_summary.json` all record exit code `0`. |
| `logged_epoch_records=200` | PASS | Read-only parsing found exactly 200 records with the complete epoch sequence 0–199. |
| `best.pt` and `last.pt` valid | PASS | Both have recorded SHA256 values, identical nonzero sizes, epochs, global steps, and formal-run commit metadata. |
| Model, optimizer, and GradScaler states exist | PASS | All three state-presence fields are `true` for both checkpoints. |
| `validation_exit_code=0` | PASS | `best_validation.log` and `run_summary.json` both record validation exit code `0`. |
| Validation record count is 385 | PASS | The CSV has exactly 385 data rows and 385 unique sample IDs. |
| PSNR and SSIM are all finite | PASS | All 385 PSNR and all 385 SSIM values parsed as finite numbers. |
| CSV means are consistent with checkpoint records | PASS | CSV means exactly match `run_summary.json`; checkpoint-selection PSNR differs by `2.7339489427902208e-05` dB, below the declared `0.0001` dB tolerance. |
| Final test was not evaluated | PASS | `train.log`, `best_validation.log`, and `run_summary.json` record `final_test_evaluated=false`; only the validation manifest was evaluated. |

All phase-specific acceptance criteria are supported by the archived evidence.

## 12. Limitations and Unresolved Risks

- This report archives one formal seed result. It does not provide a
  three-seed aggregate.
- Seed 2027 has not been completed.
- The final test remains intentionally unevaluated.
- The training-time selector metrics and independently aggregated CSV means
  have the small differences disclosed above. The PSNR difference is within
  the declared archival tolerance.
- The exact cloud shell launch commands are absent from the supplied evidence,
  although the run identity, resolved configuration, environment, manifests,
  output directory, seed, device, timing, and commits are recorded.
- This archival phase did not independently reload checkpoint files because
  training, evaluation, and checkpoint execution were prohibited.

## 13. Supported Conclusions

- The formal seed-1234 run began at epoch 0 and completed all 200 configured
  epochs, indexed 0 through 199, under formal-run commit
  `0a713111f34a3bccacecbadd6131bd641b357c7f`.
- Valid `best.pt` and `last.pt` metadata, including model, optimizer, and
  GradScaler states, is recorded.
- `best.pt` was selected at epoch 180 and global step 156865 by the highest
  formal-validation RGB PSNR rule.
- The complete 385-image formal validation succeeded, and every per-image RGB
  PSNR and SSIM value is finite.
- The formal seed-1234 validation result is mean PSNR RGB
  `27.334599338878284` and mean SSIM RGB `0.8927693091429674`.
- The final test was not evaluated.

## 14. Unsupported Conclusions and Seed-Selection Prohibition

- Seed 3407 and seed 1234 are required formal repetitions and must not be
  compared to select or report only the better seed.
- The seed-3407 and seed-1234 results cannot be used for best-seed selection,
  checkpoint selection across seeds, hyperparameter selection, or model
  selection.
- Seed 2027 has not been completed.
- No three-seed mean or standard deviation can yet be reported.
- The formal validation metrics are not final-test metrics.
- The final test has not been evaluated.
- No conclusion about three-seed stability is supported.
- No color operator or scattering operator result is supported.
- No operator-order hypothesis has been tested.
- No paper-level main experimental conclusion is supported.

## 15. Final Phase Status

**Final Phase Status: PASS**

The seed-1234 formal validation result is archived as a single-seed result
only. It must be retained together with seed 3407 and must not be used for
best-seed selection.

**Next phase authorization: false**

This report does not authorize seed 2027, final-test evaluation, or any
subsequent phase.
