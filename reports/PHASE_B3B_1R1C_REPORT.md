# Phase B3b-1r1c Report — AMP Overflow Cloud Runtime Validation

## 1. Phase Objective and Governance

- Phase: `B3b-1r1c`
- Phase name: `amp_overflow_cloud_runtime_validation`
- Objective: archive the cloud pytest, explicit CUDA AMP overflow regression, and temporary old-checkpoint resume-probe evidence for the AMP overflow remediation.
- Current phase authorized: `true`
- Next phase authorized: `false`
- Reporting-task starting commit: `abf8207877fad0e0e99f9f6717a9a92a17b3afd1`
- Ending commit: `pending`
- File changed by this archival task: `reports/PHASE_B3B_1R1C_REPORT.md`

No pytest, training, evaluation, checkpoint-resume, or final-test command was rerun during this report-only task. Source, tests, configurations, manifests, governance files, runtime evidence, and the interrupted seed-3407 run were not modified. `ORDER_STUDY_STATE.yaml` already had an uncommitted user change at the start of this task and was preserved unchanged.

## 2. Recorded Environments and Provenance

### Interrupted formal seed-3407 run

| Item | Recorded value |
| --- | --- |
| Phase | `B3b-1` |
| Seed | `3407` |
| Git commit | `d6652a2b08c323d7fb731d40faa6e6ca56a3f8cc` |
| Python | `3.12.13` |
| PyTorch | `2.13.0+cu126` |
| CUDA runtime | `12.6` |
| GPU | NVIDIA GeForce RTX 3090 |
| cuDNN | `91002` |
| Formal configuration SHA256 | `ee812ff2c6621a9e01691e016b497eaf94af152320d141b80d69209abd917e80` |
| Output directory | `/root/autodl-tmp/pro/UIE3_runs/b3b/seed_3407` |

### Cloud remediation validation

| Item | Recorded value |
| --- | --- |
| Phase | `B3b-1r1c` |
| Git commit | `46c77ed1ca8737cfbd09aed629fe9e3aa306b12a` |
| Python | `3.12.13` |
| PyTorch | `2.13.0+cu126` |
| CUDA runtime | `12.6` |
| CUDA available | `true` |
| GPU | NVIDIA GeForce RTX 3090 |
| cuDNN | `91002` |

The exact cloud commands are not present in the supplied evidence files and are therefore not reconstructed or guessed here. This report records only their committed outputs.

## 3. Original Interrupted Run

The original run used:

- seed `3407`;
- experiment name `nafnet_small_lsui_formal_seed3407`;
- canonical formal train manifest `splits/lsui19/train.tsv`;
- canonical formal validation manifest `splits/lsui19/validation.tsv`;
- NAFNet-small with the frozen formal configuration;
- Charbonnier loss with epsilon `0.001`;
- AdamW with learning rate `0.0002`, weight decay `0`, and betas `[0.9, 0.999]`;
- batch size `4`, patch size `256`, AMP enabled, and maximum `200` epochs;
- `validate_every=1` and `save_every=10`;
- no final-test evaluation.

Four epoch records completed. The last completed record was the zero-based `epoch=3` (the fourth completed epoch) at `global_step=3468`:

| Metric | Last completed value |
| --- | ---: |
| Train loss | `0.062128755473451217` |
| Validation RGB PSNR | `23.311626689465015` |
| Validation RGB SSIM | `0.8518725407588017` |

All three values are finite. Training then stopped with:

```text
FloatingPointError: Gradient for parameter 'intro.bias' contains NaN or Inf.
```

The recorded train exit code is `2`. `failure_summary.txt` identifies the status as `BLOCKED_BY_NONFINITE_GRADIENT`, the last completed `global_step` as 3468, and the reported parameter as `intro.bias`.

The original diagnostic artifacts are recorded with these hashes:

| Artifact | SHA256 |
| --- | --- |
| Original `best.pt` | `3cb30a572e8731072a46e7e2fd5d4b5b8f8fc4c0cdc791ab278ef158f258c3a7` |
| Original `last.pt` | `4d14b6ee3435d4094547c9deb5d3a4998670db8619b52a5d8948e2541f126c52` |
| Formal seed-3407 configuration | `ee812ff2c6621a9e01691e016b497eaf94af152320d141b80d69209abd917e80` |

The original checkpoints and failure evidence are preserved. This interrupted run is not a completed formal baseline and must not be used as a paper result.

## 4. Remediation Semantics

The remediation changes training-infrastructure correctness, not scientific experiment semantics:

- An AMP non-finite gradient after `GradScaler.unscale_()` no longer immediately becomes a fatal `FloatingPointError`.
- GradScaler receives the recorded overflow through `scaler.step(optimizer)` and skips the overflowed optimizer update.
- `scaler.update()` is called and the AMP scale decreases after overflow.
- A skipped optimizer update does not increment `global_step`.
- A later finite-gradient step can update parameters and continue training normally.
- In non-AMP mode, a non-finite gradient still raises `FloatingPointError`.
- The operation order remains scaled backward, unscale, optional finite-gradient clipping, scaler step, and scaler update.
- Gradient clipping remains after unscale, runs only for finite gradients, and retains `error_if_nonfinite=True`.
- The checkpoint schema and format are unchanged; `src/engine/checkpoint.py` was not changed by the remediation commit.

The remediation did not change the model architecture, loss, optimizer, learning rate, weight decay, batch size, patch size, manifests, seed, AMP setting, epoch budget, validation frequency, metrics, or checkpoint-selection rule.

## 5. Pytest Results

The exact pytest summary is:

| Item | Recorded result |
| --- | ---: |
| Passed | `54` |
| Failed | `0` |
| Warnings | `161` |
| Elapsed time | `7.26` seconds |
| Pytest exit code | `0` |

All collected tests passed. The warning total consists of one tensor-to-scalar warning and 160 `saved_variables` deprecation warnings; no warning caused test failure.

## 6. CUDA AMP Overflow Regression

### Deliberately overflowed step

`amp_cuda_regression.log` records:

| Field | Value |
| --- | --- |
| `optimizer_step_applied` | `false` |
| `amp_overflow_detected` | `true` |
| `global_step` | `0` |
| `amp_scale_before` | `65536.0` |
| `amp_scale_after` | `32768.0` |
| Loss | `0.44486355781555176` |

The scale decreased, and `global_step` did not increase. The standalone regression reached its subsequent finite step and final `PASS`, so `train_step` did not raise `FloatingPointError`. Parameter immutability for the overflowed step is also directly asserted by `test_amp_overflow_skips_optimizer_update_and_global_step`; that test was part of the 54-test passing suite.

### Subsequent finite step

The same log then records:

| Field | Value |
| --- | --- |
| `optimizer_step_applied` | `true` |
| `amp_overflow_detected` | `false` |
| `global_step` | `1` |
| `amp_scale_before` | `32768.0` |
| `amp_scale_after` | `32768.0` |
| Loss | `0.44486355781555176` |

The finite step increased `global_step` exactly once. Parameter update after recovery is directly asserted by `test_finite_amp_step_continues_after_overflow`, which also passed in the complete suite. The standalone log ends with `CUDA AMP overflow regression: PASS`.

## 7. Old-Checkpoint Compatibility and Temporary Resume Probe

| Item | Recorded value |
| --- | --- |
| Source checkpoint | `/tmp/UIE3_phase_b3b_1r1c/resume_source.pt` |
| Source SHA256 | `4d14b6ee3435d4094547c9deb5d3a4998670db8619b52a5d8948e2541f126c52` |
| Result checkpoint | `/tmp/UIE3_phase_b3b_1r1c/resume_probe/last.pt` |
| Old `global_step` | `3468` |
| Target `global_step` | `3568` |
| New `global_step` | `3568` |
| Successful optimizer updates added | `100` |
| Model state present | `true` |
| Optimizer state present | `true` |
| GradScaler state present | `true` |
| Resume-probe exit code | `0` |
| Probe status | `PASS` |
| `formal_training_resumed` | `false` |
| `final_test_evaluated` | `false` |

The temporary source SHA256 exactly matches the recorded original `last.pt` SHA256. The repaired trainer's strict resume path restored the old checkpoint and completed the temporary target of 100 additional successful optimizer updates. The result checkpoint was written only under `/tmp/UIE3_phase_b3b_1r1c/`.

`resume_probe.log` reports formal CLI mode because the compatibility probe intentionally used the frozen formal configuration and canonical train/validation manifests. The machine-readable summary nevertheless explicitly declares `formal_training_resumed=false` and identifies its purpose as temporary checkpoint compatibility and real-data resume validation only. It is neither continuation of the official formal run nor a formal experiment result.

## 8. Scientific Semantics and Experimental Validity

- AMP overflow handling is a training-infrastructure correctness repair.
- The repair does not change the model, dataset, loss, optimizer, learning rate, formal seed, metrics, validation schedule, or checkpoint-selection rule.
- The interrupted pre-fix run and any post-fix run must not be concatenated into a final paper result.
- A new formal seed-3407 run must start from `epoch 0` under one fixed repaired Git commit.
- The new formal run must use a new output directory.
- The original `/root/autodl-tmp/pro/UIE3_runs/b3b/seed_3407` directory must remain diagnostic evidence only.
- The temporary resume probe and its validation metrics are compatibility evidence, not a baseline result.
- Final test was not evaluated.

## 9. Acceptance Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| All pytest tests passed | **PASS** | 54 passed, 0 failed, exit code 0. |
| CUDA AMP overflow was skipped correctly | **PASS** | `optimizer_step_applied=false`, `amp_overflow_detected=true`, standalone regression PASS. |
| Parameters did not update during overflow | **PASS** | Explicit regression-test assertion passed in the complete pytest suite. |
| `global_step` did not increase during overflow | **PASS** | Overflow result remained at `global_step=0`. |
| AMP scale decreased | **PASS** | `65536.0 → 32768.0`. |
| A later finite step continued successfully | **PASS** | Successful result reached `global_step=1`; recovery parameter-update assertion passed. |
| Strict non-AMP behavior remains | **PASS** | `test_non_amp_nonfinite_gradient_remains_fatal` passed. |
| Old checkpoint restored successfully | **PASS** | Strict temporary resume probe exited 0 and reached its target. |
| Model, optimizer, and scaler states exist | **PASS** | All three summary flags are true. |
| Temporary resume probe succeeded | **PASS** | Summary status PASS; 100 successful updates added. |
| Formal configuration remained unchanged | **PASS** | Remediation changed trainer/tests/report only; recorded formal configuration hash is preserved. |
| Final test was not evaluated | **PASS** | `final_test_evaluated=false`. |
| Formal seed 3407 remains unresumed/unrestarted | **PASS** | `formal_training_resumed=false`; only the isolated `/tmp` compatibility probe ran. |

All acceptance criteria are supported by the supplied runtime evidence.

## 10. Supported Conclusion

“AMP overflow处理修复已通过云端pytest、真实CUDA overflow回归和旧checkpoint临时恢复验证。训练器能够跳过溢出的optimizer更新、正确维护global_step并继续后续有限更新。seed 3407可在单独人工授权后，使用修复后的单一Git commit和新输出目录从epoch 0重新正式训练。”

## 11. Unsupported Conclusions

- 尚未完成修复后的seed 3407正式训练；
- 原中断运行不是正式baseline结果；
- 尚未得到seed 3407最终validation结果；
- 尚未运行seed 1234和2027；
- 尚未评估final test；
- 尚未获得三seed均值和方差；
- 尚未实现颜色算子；
- 尚未实现散射算子；
- 尚未验证算子顺序假设。

## 12. Final Phase Status

**Final Phase Status: PASS**

**next phase authorization=false**

Phase B3b-1r1c stops here. No formal seed training or next phase is authorized by this report.
