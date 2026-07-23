# Phase B3b-1r2 Formal Seed 3407 Result Report

## 1. Phase Objective and Governance

This phase archives the completed formal NAFNet-small baseline training and
formal-validation result for seed 3407. It does not modify or rerun training,
evaluation, source code, configuration, manifests, governance files, or runtime
evidence.

- Protocol version: `1.0.1`
- Phase ID: `B3b-1r2`
- Phase name: `baseline_formal_training_seed_3407_restart`
- Current phase authorized: `true`
- Next phase authorized: `false`
- Required report:
  `reports/PHASE_B3B_SEED3407_R1_REPORT.md`
- Final test evaluated: `false`
- Seed 1234 authorized by this report: `false`

No conflict was found among the protocol, decision log, current state, and the
phase-specific archival instructions.

## 2. Commits and Run Identity

- Formal-run Git commit:
  `79682aafff31fec602921acc63ad07faa9710938`
- Git commit recorded by the running code, `best.pt`, and `last.pt`:
  `79682aafff31fec602921acc63ad07faa9710938` in all three cases
- Formal-run code policy: one fixed, remediated Git commit for the complete run
- Run origin: `epoch_0_restart`
- Interrupted pre-remediation run resumed: `false`
- Seed: `3407`
- Formal configuration:
  `configs/nafnet_small_lsui_formal_seed3407.yaml`
- Configuration SHA256:
  `ee812ff2c6621a9e01691e016b497eaf94af152320d141b80d69209abd917e80`
- Formal train manifest: `splits/lsui19/train.tsv`
- Formal validation manifest: `splits/lsui19/validation.tsv`
- New output directory:
  `/root/autodl-tmp/pro/UIE3_runs/b3b/seed_3407_r1`
- Repository HEAD at the start of report authoring:
  `677b9c42e13ca6258fe8a2f41e63520eeaad93a6`
- Ending report commit: pending; this report has not been committed

The run started again at epoch 0 in the new output directory. It did not
restore or append to the interrupted pre-remediation seed-3407 run.

## 3. Changed Files

The only file created in this archival phase is:

- `reports/PHASE_B3B_SEED3407_R1_REPORT.md`

No source, test, configuration, manifest, governance, runtime-evidence, or
dataset file was modified.

## 4. Assumptions and Evidence Boundary

- Runtime facts are taken only from the files required by the phase prompt.
- Checkpoint validity is assessed from the recorded SHA256, nonzero size, epoch,
  global step, and state-presence fields in `run_summary.json`. The checkpoints
  were not loaded again in this report-only phase.
- The exact original shell launch command is not present in the supplied
  evidence. `train.log` records that the process was run under `nohup`, together
  with the complete resolved configuration, run mode, manifests, output
  directory, seed, device, and Git commit. No unrecorded command is inferred.
- The independent best-checkpoint validation CSV was parsed read-only with the
  Python standard library. No model code or dataset image was accessed.

## 5. Commands

No training, evaluation, test, or final-test command was run in this archival
phase. The report was prepared with read-only inspection commands:

- `sed` and `cat` for the required governance, configuration, log, CSV, and JSON
  evidence;
- `rg` for the current-phase authorization and required-output entries;
- `git status`, `git branch --show-current`, and `git rev-parse HEAD`;
- `shasum -a 256` and `wc -l` for local evidence-file checks;
- a Python-standard-library-only read-only parser for epoch records, CSV row
  count, finite-value checks, means, and cross-file consistency.

The exact formal training and validation launch commands are not reproduced
because they are absent from the supplied evidence.

## 6. Environment and Fixed Formal Configuration

The archived cloud environment records:

- Python: `3.12.13`
- PyTorch: `2.13.0+cu126`
- CUDA runtime: `12.6`
- cuDNN: `91002`
- GPU: `NVIDIA GeForce RTX 3090`
- Device used by training: `cuda`
- Model parameter count: `2,846,755`
- Run mode: `FORMAL`
- Formal experiment flag: `true`

The resolved configuration in `train.log` records the frozen NAFNet-small
model, seed 3407, formal train and validation manifests, patch size 256, batch
size 4, four workers, the fixed augmentations, Charbonnier loss with epsilon
`0.001`, AdamW with learning rate `0.0002`, weight decay `0`, AMP enabled, 200
epochs, validation every epoch, and periodic checkpoint retention every 10
epochs. No test manifest is present.

## 7. Formal Training Result

- Formal run start: `2026-07-23T00:52:14+08:00`
- Formal run end: `2026-07-23T07:46:53+08:00`
- Total training elapsed time: `24,879` seconds (`6 h 54 min 39 s`)
- Training exit code: `0`
- Logged epoch records: `200`
- Completed epoch range: `0` through `199`
- Actual completed epoch count: `200`
- Epoch sequence complete: `true`
- All logged train-loss, validation-PSNR, and validation-SSIM values finite:
  `true`

The 200-epoch formal training budget was completed. AMP overflow events were
handled by the remediated trainer without terminating the run; `global_step`
therefore represents successful optimizer updates rather than attempted
minibatches.

## 8. Checkpoints and Selection Rule

Checkpoint selection used the frozen rule:

> highest formal-validation RGB PSNR

| Checkpoint | Epoch | Global step | Size (bytes) | SHA256 |
| --- | ---: | ---: | ---: | --- |
| `best.pt` | 192 | 167,267 | 34,523,299 | `81d15252045ff96bf3c241a580ec35076eea035b7bdafe9c8cb1393c827fdd81` |
| `last.pt` | 199 | 173,334 | 34,523,299 | `6784564c41728993839808465ffc97b3d97acf615af06787e7a2e495201782cd` |

For both checkpoints, `run_summary.json` records:

- model state present: `true`;
- optimizer state present: `true`;
- GradScaler state present: `true`;
- the same formal-run Git commit:
  `79682aafff31fec602921acc63ad07faa9710938`.

The training-time selector record for epoch 192 is:

- validation RGB PSNR: `27.319525305017248`;
- validation RGB SSIM: `0.8942947139987698`;
- `is_best: true`.

`best.pt` and `last.pt` are therefore recorded as nonempty, hashed checkpoints
with complete model, optimizer, and scaler state.

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
- Every row contains `psnr_rgb` and `ssim_rgb`: `true`
- All per-image PSNR and SSIM values finite: `true`
- Formal mean RGB PSNR: `27.319515354602366`
- Formal mean RGB SSIM: `0.8942945567044345`
- Final test evaluated: `false`

The means recomputed directly from the 385 CSV rows exactly match
`run_summary.json`:

- PSNR absolute delta: `0.0`;
- SSIM absolute delta: `0.0`.

The training-time checkpoint-selection record and the independent per-image CSV
evaluation are numerically consistent but not bit-for-bit identical:

- RGB PSNR absolute delta: `9.950414881387815e-06`;
- RGB SSIM absolute delta: `1.5729433533451243e-07`.

The evidence does not state the cause of these small differences, so this
report does not infer one. The selected checkpoint is epoch 192 in both the
training log and checkpoint metadata, and the formal reported metrics above are
the independently archived 385-image CSV means.

This is the formal validation result for seed 3407. It is not a final-test
result and is not the final three-seed baseline summary.

## 10. Actual Outputs

- `train.log` ends with `TRAIN_EXIT_CODE=0`.
- `train_exit_code.txt` records `train_exit_code=0`.
- `run_summary.json` records `FORMAL_SEED_RUN_COMPLETED`.
- `best_validation.log` records `validation_exit_code=0`,
  `num_samples=385`, and `final_test_evaluated=false`.
- `best_validation_metrics.csv` contains one header and 385 data rows.
- The configuration file's locally verified SHA256 exactly matches the hash
  recorded in the runtime environment and run summary.
- No final-test metric or final-test CSV was produced by the recorded run.

## 11. Acceptance Criteria

| Criterion | Result | Evidence |
| --- | --- | --- |
| `train_exit_code=0` | PASS | Both training exit-code records are `0`. |
| Formal training budget completed | PASS | Exactly 200 epoch records cover epochs 0–199. |
| `best.pt` and `last.pt` valid | PASS | Both have recorded SHA256 values, identical nonzero sizes, epochs, global steps, and state metadata. |
| Checkpoint state complete | PASS | Model, optimizer, and scaler state are recorded present for both checkpoints. |
| Complete validation successful | PASS | Validation exit code is `0` for the formal validation manifest. |
| CSV contains 385 records | PASS | Read-only parsing found exactly 385 data rows and 385 unique sample IDs. |
| PSNR and SSIM all finite | PASS | All 770 metric values parsed as finite numbers. |
| Checkpoint record and CSV mean consistent | PASS | CSV means exactly equal the run summary; training-time selector versus CSV deltas are `9.950414881387815e-06` PSNR and `1.5729433533451243e-07` SSIM, disclosed above. |
| Old interrupted run not resumed | PASS | Run origin is `epoch_0_restart`; resume flags are `false`; the new output directory is used. |
| Final test not evaluated | PASS | Both validation log and run summary record `final_test_evaluated=false`; only the validation manifest was evaluated. |

All phase-specific acceptance criteria are supported by the archived evidence.

## 12. Unresolved Risks

- This report contains one formal seed only. Seeds 1234 and 2027 have not been
  completed, so three-seed variability is unknown.
- The final test remains intentionally unevaluated.
- The training-time selector metrics and the independent CSV means have the
  small, explicitly reported numerical differences above. The supplied
  evidence does not document their cause.
- The exact cloud shell launch command was not included in the supplied runtime
  evidence. The resolved configuration and run identity are complete, but the
  missing literal command is retained as a reproducibility-metadata gap.
- This archival phase did not independently reload checkpoint files because
  rerunning model or evaluation code was prohibited.

## 13. Supported Conclusions

- The remediated, single-commit seed-3407 formal run restarted from epoch 0 and
  completed all 200 configured epochs without resuming the interrupted run.
- Valid `best.pt` and `last.pt` checkpoint metadata, including model, optimizer,
  and scaler states, was recorded.
- `best.pt` was selected at epoch 192 by the highest formal-validation RGB PSNR
  rule.
- The complete 385-image formal validation succeeded, and every per-image RGB
  PSNR and SSIM value is finite.
- The formal seed-3407 validation result is mean RGB PSNR
  `27.319515354602366` and mean RGB SSIM `0.8942945567044345`.
- The final test was not evaluated.

## 14. Unsupported Conclusions

- This is not the final three-seed baseline summary.
- No three-seed mean or standard deviation can be reported.
- Seeds 1234 and 2027 have not been completed or accepted.
- The formal validation metrics are not final-test metrics.
- The final test has not been evaluated.
- No conclusion about cross-seed stability is supported.
- No color operator or scattering operator result is supported.
- No operator-order hypothesis has been tested.
- No paper-level main experimental conclusion is supported.

## 15. Final Phase Status

**PASS**

The seed-3407 formal validation result is archived as a single-seed result only.
It must not be represented as a final-test result or as a three-seed aggregate.

**Next phase authorization: false**

This report does not authorize seed 1234, seed 2027, final-test evaluation, or
any subsequent phase.
