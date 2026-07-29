# Phase B3b-3 Formal Seed 2027 Result Archive Report

## 1. Phase Objective and Governance

This report-only step archives the completed formal NAFNet-small baseline
training and formal-validation result for seed 2027. It does not modify or
rerun training, evaluation, source code, tests, configuration, manifests,
governance files, runtime evidence, checkpoints, or dataset files.

- Protocol version: `1.0.1`
- Phase ID: `B3b-3`
- Phase name: `baseline_formal_training_seed_2027`
- Current phase authorized: `true`
- Next phase authorized: `false`
- Required report: `reports/PHASE_B3B_SEED2027_REPORT.md`
- Final test evaluated: `false`

The current state satisfies the phase-specific authorization conditions:
`current_phase.id == B3b-3`, `current_phase.authorized == true`, and
`current_phase.next_phase_authorized == false`.

## 2. Formal Run Identity and Commits

- Seed: `2027`
- Formal-run Git commit:
  `f0a802e2bbd82ed8621c8f01087d36d8c09d680e`
- Git commit recorded by the runtime environment:
  `f0a802e2bbd82ed8621c8f01087d36d8c09d680e`
- Git commit recorded by `best.pt`:
  `f0a802e2bbd82ed8621c8f01087d36d8c09d680e`
- Git commit recorded by `last.pt`:
  `f0a802e2bbd82ed8621c8f01087d36d8c09d680e`
- Repository HEAD at the start of report authoring:
  `f79b4b5a468d9054bb0dbfad0283f9586d2e464d`
- Ending report commit: pending; this report has not been committed
- Run origin: `epoch_0`
- Formal configuration:
  `configs/nafnet_small_lsui_formal_seed2027.yaml`
- Configuration SHA256:
  `e82e17d95822da8b64e5439c3bdf8bc75defd015fa1c06588c26e4ad8fc024dc`
- Formal train manifest: `splits/lsui19/train.tsv`
- Formal validation manifest: `splits/lsui19/validation.tsv`
- Output directory:
  `/root/autodl-tmp/pro/UIE3_runs/b3b/seed_2027`

The runtime environment, training log, `best.pt`, and `last.pt` all record the
same formal-run Git commit. The training log records
`RUN_ORIGIN=EPOCH_0`, begins with epoch 0, and continues through epoch 199.

## 3. Changed Files

The only file created in this archival step is:

- `reports/PHASE_B3B_SEED2027_REPORT.md`

No source, test, configuration, manifest, governance, runtime-evidence,
checkpoint, or dataset file was modified.

## 4. Evidence Boundary

- Runtime facts are taken from the configuration, environment, training log,
  validation log, validation CSV, and `run_summary.json` required by the
  phase-specific prompt.
- Checkpoint validity is assessed from the recorded nonzero size, SHA256,
  epoch, global step, formal-run commit, and state-presence fields in
  `run_summary.json`. The checkpoint files were not reloaded in this
  report-only step.
- The independent validation CSV was parsed read-only with the Python standard
  library. No model code, checkpoint, dataset image, training, or evaluation
  path was executed.
- The literal cloud launch command is not present in the supplied evidence.
  The logs record the resolved run identity and configuration; no missing
  command is inferred.

## 5. Read-Only Archival Checks

No training, validation, test, or final-test command was run while preparing
this report. The report used only read-only inspection and parsing:

- the required governance, prior reports, configuration, environment, logs,
  CSV, and JSON evidence were read;
- the current phase authorization and repository state were checked;
- all epoch JSON records were parsed for count, continuity, and finite values;
- the validation CSV was parsed for schema, record count, unique sample IDs,
  finite metrics, and independent PSNR and SSIM means;
- checkpoint metadata, commits, hashes, state-presence flags, and metric
  consistency were checked across the real logs and `run_summary.json`.

## 6. Environment and Frozen Formal Configuration

The archived runtime environment records:

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

The resolved training configuration records seed 2027, the frozen
NAFNet-small structure, formal train and validation manifests, patch size 256,
batch size 4, four workers, horizontal and vertical flips, 90-degree rotation,
Charbonnier loss with epsilon `0.001`, AdamW with learning rate `0.0002`,
weight decay `0`, AMP enabled, 200 epochs, validation every epoch, and
periodic checkpoint retention every 10 epochs. No test manifest is present.

The locally recomputed configuration SHA256 exactly matches the value recorded
in `environment.txt` and `run_summary.json`.

## 7. Formal Training Completed Epochs 0–199

- Formal run start: `2026-07-29T00:38:51+08:00`
- Formal run end: `2026-07-29T07:29:25+08:00`
- Total training elapsed time: `24,634` seconds (`6 h 50 min 34 s`)
- Training exit code: `0`
- Logged epoch records: `200`
- Completed epoch range: `0` through `199`
- Actual completed epoch count: `200`
- Epoch sequence complete: `true`
- All logged train-loss, validation-PSNR, validation-SSIM,
  best-validation-PSNR, and learning-rate values finite: `true`

The formal run started from epoch 0 and completed the full frozen 200-epoch
budget. Handled AMP overflow events are present in the log and did not
terminate the run.

## 8. Checkpoints and Selection Rule

Checkpoint selection used the frozen rule:

> highest formal-validation RGB PSNR

| Checkpoint | Epoch | Global step | Size (bytes) | SHA256 |
| --- | ---: | ---: | ---: | --- |
| `best.pt` | 188 | 163,799 | 34,523,299 | `29814cd2c0ab7e893d0bd329ded2c9875840ab8c51d0163e412a63297023c0f0` |
| `last.pt` | 199 | 173,334 | 34,523,299 | `3fda66ab0eaa4449d499fd05834533465e8aa8231b5239aa534ceedc037fb372` |

For both checkpoints, `run_summary.json` records:

- model state present: `true`;
- optimizer state present: `true`;
- GradScaler state present: `true`;
- nonzero checkpoint size: `true`;
- a valid 64-character SHA256: `true`;
- the formal-run Git commit:
  `f0a802e2bbd82ed8621c8f01087d36d8c09d680e`.

The training-time checkpoint-selection record at epoch 188 is:

- checkpoint-selection RGB PSNR: `27.03952445736179`;
- validation RGB SSIM: `0.8912330667693894`;
- global step: `163799`;
- `is_best: true`.

The epoch-188 validation PSNR is the highest of all 200 training-log
validation records. `best.pt` and `last.pt` therefore have complete recorded
checkpoint metadata and complete model, optimizer, and GradScaler state.

## 9. Complete Independent Formal Validation

The archived post-training evaluation used `best.pt` and the formal validation
manifest.

- Validation split: `validation`
- Validation exit code: `0`
- Validation elapsed time: `21` seconds
- Formal-validation CSV records: `385`
- Unique CSV sample IDs: `385`
- CSV columns:
  `sample_id`, `input_relative_path`, `gt_relative_path`, `psnr_rgb`,
  `ssim_rgb`
- All 385 per-image PSNR values finite: `true`
- All 385 per-image SSIM values finite: `true`
- Independent CSV Mean PSNR RGB: `27.039529166283547`
- Independent CSV Mean SSIM RGB: `0.8912330272909883`
- Checkpoint-selection PSNR: `27.03952445736179`
- CSV/checkpoint PSNR absolute difference:
  `4.708921757412554e-06` dB
- Declared archival consistency tolerance: `1e-4` dB
- Difference within the `1e-4` dB tolerance: `true`
- Final test evaluated: `false`

The PSNR and SSIM means were independently recomputed from all 385 CSV rows
and exactly match `run_summary.json`. The checkpoint-selection PSNR and
independent CSV Mean PSNR RGB are not bit-for-bit identical, but their
absolute difference is below the declared archival tolerance:

\[
4.708921757412554\times10^{-6}\ \mathrm{dB}
<
1.0\times10^{-4}\ \mathrm{dB}.
\]

The supplied evidence does not establish the cause of the small numerical
difference, so this report does not infer one.

This is the formal validation result for seed 2027. It is not a final-test
result.

## 10. Actual Runtime Outputs

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
| `train_exit_code=0` | PASS | `train.log`, `train_exit_code.txt`, and `run_summary.json` record exit code `0`. |
| `logged_epoch_records=200` | PASS | Exactly 200 epoch records form the complete sequence 0–199. |
| `best.pt` and `last.pt` valid | PASS | Both have recorded nonzero sizes, valid SHA256 values, epochs, global steps, and formal-run commit metadata. |
| Checkpoint state complete | PASS | Model, optimizer, and GradScaler state-presence fields are `true` for both checkpoints. |
| `validation_exit_code=0` | PASS | `best_validation.log` and `run_summary.json` record validation exit code `0`. |
| Formal validation sample count is 385 | PASS | The CSV has 385 rows and 385 unique sample IDs. |
| All PSNR and SSIM values finite | PASS | All 385 PSNR and all 385 SSIM values parsed as finite numbers. |
| CSV and checkpoint PSNR consistent within `1e-4` dB | PASS | Absolute difference is `4.708921757412554e-06` dB. |
| Final test not evaluated | PASS | Runtime evidence records `final_test_evaluated=false`; only formal validation was evaluated. |

All phase-specific acceptance criteria are satisfied by the archived evidence.

## 12. Limitations and Unresolved Work

- This report archives the formal validation result for seed 2027 only.
- Seed 3407, seed 1234, and seed 2027 are mandatory formal repetitions. They
  must not be compared to choose one seed as the final baseline.
- The three-seed mean and standard deviation are not calculated in this phase.
- The final test remains unevaluated.
- The checkpoint files were not reloaded during report preparation because
  checkpoint execution and rerunning evaluation were prohibited.
- The literal original cloud launch command is absent from the supplied
  evidence, although the resolved run identity, environment, configuration,
  manifests, output directory, seed, device, timing, and commit are recorded.

## 13. Supported Conclusions

- The formal seed-2027 run started at epoch 0 and completed all 200 configured
  epochs, indexed 0 through 199, under formal-run commit
  `f0a802e2bbd82ed8621c8f01087d36d8c09d680e`.
- `best.pt` was selected at epoch 188 and global step 163799 by the highest
  formal-validation RGB PSNR rule.
- `best.pt` and `last.pt` have complete recorded model, optimizer, and
  GradScaler state.
- The complete 385-image formal validation succeeded and every per-image RGB
  PSNR and SSIM value is finite.
- The independent formal-validation result for seed 2027 is Mean PSNR RGB
  `27.039529166283547` and Mean SSIM RGB `0.8912330272909883`.
- The checkpoint-selection PSNR and CSV mean are consistent within `1e-4` dB.
- The final test was not evaluated.

## 14. Unsupported Conclusions and Seed-Selection Prohibition

- The three formal seeds must not be compared to select or report only the
  best-performing seed as the final baseline.
- No cross-seed checkpoint selection, hyperparameter selection, model
  selection, or best-seed selection is authorized.
- The three-seed validation mean and standard deviation have not been
  calculated in this phase.
- The formal validation metrics are not final-test metrics.
- The final test has not been evaluated.
- No final baseline aggregate or final-test conclusion is supported.
- No color operator, scattering operator, operator-order, Oracle, or routing
  conclusion is supported.

## 15. Final Phase Status

**Final Phase Status: PASS**

This report archives the completed formal validation result for seed 2027.
It does not select a preferred seed, compute the three-seed aggregate, or
evaluate the final test.

**Next phase authorization: false**

This report does not authorize the next phase, three-seed aggregation,
final-test evaluation, or any subsequent work.
