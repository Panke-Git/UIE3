# Phase B3a Report — Formal Baseline Configuration and Resource Validation

## 1. Phase Scope and Governance

- Phase: `B3a`
- Phase name: `baseline_formal_config_freeze_and_resource_validation`
- Objective: archive the recorded 100-step resource probe and complete-validation evidence, then make an evidence-bounded feasibility decision for the NAFNet-small formal baseline candidate configuration.
- Current phase authorized: `true`
- Next phase authorized: `false`
- This report does not authorize Phase B3b and does not record a formal baseline result.
- Reporting-task starting repository commit: `0c0b7c50dfbdb0f942e56a2c7959923941d4ade0`
- Cloud runtime Git commit: `17b2847e9a1f3a8b841172e16614b9c1e94dad0a`
- Ending commit: `pending` (this report has not been committed as part of this archival task)
- Changed file for this task: `reports/PHASE_B3A_REPORT.md`

## 2. Evidence and Commands

The decision in this report is limited to the following recorded evidence:

- `reports/runtime/phase_b3a/environment.txt`
- `reports/runtime/phase_b3a/resource_probe.log`
- `reports/runtime/phase_b3a/resource_probe.json`
- `reports/runtime/phase_b3a/full_validation.log`
- `reports/runtime/phase_b3a/full_validation_metrics.csv`
- `configs/nafnet_small_lsui_formal.yaml`
- `splits/lsui19/SPLIT_METADATA.json`

Recorded resource-probe command:

```text
/root/miniconda3/envs/pax/bin/python tools/train_baseline.py --config configs/nafnet_small_lsui_formal.yaml --dataset-root /root/autodl-tmp/pro/publicdata/LSUI19_dup_train --device cuda --output-dir /tmp/UIE3_phase_b3a/resource_probe --max-steps 100 --num-workers 4
```

Recorded full-validation command:

```text
/root/miniconda3/envs/pax/bin/python tools/evaluate_baseline.py --config configs/nafnet_small_lsui_formal.yaml --dataset-root /root/autodl-tmp/pro/publicdata/LSUI19_dup_train --checkpoint /tmp/UIE3_phase_b3a/resource_probe/last.pt --split validation --device cuda --output-csv reports/runtime/phase_b3a/full_validation_metrics.csv --num-workers 4
```

No training, evaluation, or test command was rerun during report archival. Read-only checks confirmed that the candidate configuration SHA256 matches both runtime evidence files, the validation CSV contains 385 data rows with finite metrics, and the runtime-evidence commit changes only the five files under `reports/runtime/phase_b3a/`.

## 3. Environment and Data

| Item | Recorded value |
| --- | --- |
| Python | `3.12.13` (Anaconda build; GCC 14.3.0) |
| PyTorch | `2.13.0+cu126` |
| CUDA runtime | `12.6` |
| CUDA available | `true` |
| cuDNN | `91002` |
| GPU | `NVIDIA GeForce RTX 3090` |
| GPU count | `1` |
| Cloud runtime Git commit | `17b2847e9a1f3a8b841172e16614b9c1e94dad0a` |
| Formal configuration SHA256 | `6eadaed31a60127930c302f2fb11a554952fb759bb7c04fdf9110b88cd70755d` |
| Dataset root | `/root/autodl-tmp/pro/publicdata/LSUI19_dup_train` |
| Formal train samples | `3466` |
| Formal validation samples | `385` |
| Final test loaded/evaluated | `false` |

The computed SHA256 of `configs/nafnet_small_lsui_formal.yaml` matches the hash recorded in both `environment.txt` and `resource_probe.json`. `SPLIT_METADATA.json` independently records formal counts of 3466 train, 385 validation, and 428 held-out test samples with dataset audit status `PASS`. Runtime logs resolve only the canonical train and validation manifests; `resource_probe.json` records `final_test_evaluated: false`.

## 4. Candidate Formal Configuration

| Category | Candidate value |
| --- | --- |
| Model | NAFNet-small |
| `img_channel` | `3` |
| `width` | `32` |
| `enc_blk_nums` | `[2, 2, 2]` |
| `middle_blk_num` | `4` |
| `dec_blk_nums` | `[2, 2, 2]` |
| Patch size | `256` |
| Batch size | `4` |
| Data-loader workers | `4` |
| AMP | `true` |
| Loss | Charbonnier |
| Charbonnier epsilon | `1.0e-3` |
| Optimizer | AdamW |
| Learning rate | `2.0e-4` |
| Weight decay | `0` |
| Adam betas | `[0.9, 0.999]` |
| Candidate epochs | `200` |
| Checkpoint selection | Highest formal-validation RGB PSNR |
| Seed template | `3407` |
| Later formal seed set | `[3407, 1234, 2027]` |

The final test manifest is excluded from checkpoint selection. The configuration uses no pretrained weights, extra loss, EMA, GAN, diffusion model, or external prior. The 200-epoch value remains a candidate resource-planning value; it was not executed in Phase B3a.

## 5. 100-Step Resource Probe

| Item | Recorded result |
| --- | --- |
| `RUN_MODE` | `FORMAL` |
| `FORMAL_EXPERIMENT` | `true` |
| Completed global step | `100` |
| Checkpoint global step | `100` |
| Train loss | `0.09442494377493858` |
| Non-finite loss | Not observed; the recorded loss is finite |
| CUDA OOM | Not observed |
| Data-loader worker error | Not observed |
| Total wall time | `60.162727` seconds |
| Estimated training-only duration | `39.429545` seconds |
| Estimated seconds per train step | `0.39429545` |
| Estimated training throughput | `10.144677043572276` images/second |
| Peak sampled GPU memory | `17148.0` MiB |
| GPU telemetry samples | `67` |
| Checkpoint path | `/tmp/UIE3_phase_b3a/resource_probe/last.pt` |
| Checkpoint size | `34523299` bytes |
| `model_state_dict` present | `true` |
| `optimizer_state_dict` present | `true` |
| Training exit code | `0` |

This run was only a 100-step resource probe. `FORMAL_EXPERIMENT=true` means that the formal candidate configuration and canonical formal manifests were used; it does **not** mean that formal training was completed. The training log also reports validation PSNR `19.991800446943802` and SSIM `0.771301347249514`, but these are pipeline-validation observations from the 100-step checkpoint, not formal baseline results.

## 6. Complete Validation Pipeline

| Item | Recorded result |
| --- | --- |
| Checkpoint | `/tmp/UIE3_phase_b3a/resource_probe/last.pt` |
| Split | `validation` |
| Samples | `385` |
| Process exit code | `0` |
| Elapsed time | `20.733182` seconds |
| CSV columns | `sample_id`, `input_relative_path`, `gt_relative_path`, `psnr_rgb`, `ssim_rgb` |
| CSV data rows | `385` |
| Every row has PSNR and SSIM | `true` |
| All PSNR values finite | `true` |
| All SSIM values finite | `true` |
| Mean RGB PSNR | `19.991820521168894` |
| Mean RGB SSIM | `0.7712961410547232` |
| Final test loaded/evaluated | `false` |

The two means above were recomputed from the 385 CSV rows and match the values in `full_validation.log` and `resource_probe.json`.

**PIPELINE_VALIDATION_ONLY**

**NOT_A_FORMAL_BASELINE_RESULT**

## 7. Training-Cost Estimate

The following values are reproduced from `resource_probe.json`:

| Field | Value |
| --- | ---: |
| `estimated_seconds_per_train_step` | `0.39429545` |
| `estimated_steps_per_epoch` | `867` |
| `estimated_train_only_epoch_seconds` | `341.85415515` |
| `estimated_total_epoch_seconds` | `362.58733715` |
| `estimated_total_epoch_minutes` | `6.043122285833333` |
| `estimated_200_epoch_hours` | `20.14374095277778` |
| `validation_elapsed_seconds` | `20.733182` |

Limitations:

- The estimate is based on only 100 training steps.
- Startup and checkpoint-writing overhead remain included.
- The training-run validation and standalone validation can have different overhead.
- GPU telemetry was sampled once per second and may miss a brief memory peak.
- The 200-epoch estimate is not an observed completion time.
- Long-duration training stability has not been tested.
- The total cost of the three formal seeds has not been validated.
- The measurements were made on an RTX 3090 and cannot be transferred to the protocol's RTX 3060 12 GB primary target without a new resource check.

## 8. Configuration Freeze Decision

| Candidate item | Decision | Evidence-bounded rationale |
| --- | --- | --- |
| `batch_size=4` | **REVISE** | The probe did not OOM on the RTX 3090, but its sampled peak was `17148.0` MiB. This exceeds the protocol's RTX 3060 12 GB primary target capacity and therefore does not leave a reasonable target-hardware margin. A lower batch size must be selected and re-probed on the target or equivalently constrained hardware before formal training. |
| `patch_size=256` | **FREEZE** | The complete 100-step CUDA probe reached global step 100 with finite loss and no recorded shape, worker, or runtime failure. Keep patch size 256 while revising batch size. |
| `num_workers=4` | **FREEZE** | The 100-step training path and the complete 385-sample validation path both exited successfully with no recorded worker error. |
| `AMP=true` | **FREEZE** | AMP was enabled in the resolved configuration and the recorded CUDA run completed without a recorded AMP exception, non-finite loss, or CUDA OOM. |
| AdamW, learning rate `2.0e-4` | **FREEZE** | These values are frozen by protocol for the candidate. This decision is not a claim that optimizer or learning rate was performance-optimized. |
| `epochs=200` | **UNRESOLVED** | The measured estimate is `20.14374095277778` hours for one 200-epoch run on the probed RTX 3090, but long-run stability, target-hardware cost, and three-seed cost were not validated. The result does not establish that 200 epochs is performance-optimal. |

The mandatory batch-size revision prevents the candidate configuration from being frozen as written. No configuration file is modified in this phase; the revision requires later human authorization and a new target-constrained resource probe.

## 9. Acceptance Criteria

| Criterion | Decision | Evidence |
| --- | --- | --- |
| Formal train manifest loaded successfully | **PASS** | Runtime resolved `splits/lsui19/train.tsv`; metadata and environment record 3466 train samples. |
| Formal validation manifest loaded successfully | **PASS** | Runtime resolved `splits/lsui19/validation.tsv`; standalone validation processed 385 samples. |
| 100 optimizer steps completed | **PASS** | Probe and checkpoint both record `global_step=100`; training exit code is 0. |
| `global_step=100` | **PASS** | Recorded in both the training log and resource JSON. |
| Loss is finite | **PASS** | `train_loss=0.09442494377493858`. |
| No CUDA OOM | **PASS** | No OOM is recorded and the CUDA process exits with code 0. |
| Checkpoint saved successfully | **PASS** | `last.pt` is recorded at 34,523,299 bytes with checkpoint global step 100. |
| Model and optimizer states exist | **PASS** | `model_state_present=true`; `optimizer_state_present=true`. |
| Complete 385-image validation succeeded | **PASS** | Process exit code 0 and CSV contains exactly 385 data rows. |
| Every per-image PSNR and SSIM is finite | **PASS** | Read-only CSV validation confirms all 385 `psnr_rgb` and `ssim_rgb` values are finite. |
| Final test was not evaluated | **PASS** | `final_test_evaluated=false`; runtime commands and resolved manifests use only train and validation. |
| Resource statistics are complete | **PASS** | Step time, throughput, wall time, memory, checkpoint size, epoch estimates, and validation time are recorded. |
| No source, manifest, or dataset was modified by the runtime validation | **PASS** | The runtime-evidence commit changes only five files under `reports/runtime/phase_b3a/`; the recorded commands contain no dataset mutation, and no source or manifest change is present. |

All runtime-chain acceptance criteria pass. The separate requirement that the candidate contain no mandatory configuration revision does not pass because `batch_size=4` is infeasible for the protocol's 12 GB primary target based on the measured 17,148 MiB peak.

## 10. Unresolved Risks and Required Follow-up

- A lower batch size has not yet been selected or measured under a 12 GB GPU constraint.
- Peak GPU memory is sampled telemetry and may understate a short-lived peak.
- Changing batch size can change optimization behavior; any later adjustment must be recorded and consistently applied across formal seeds.
- The 200-epoch estimate has not been verified through a long run and does not establish convergence or optimal stopping.
- Only seed 3407 was used for this resource probe; three-seed stability and total cost remain unknown.
- The small difference between the training-loop validation summary and the standalone CSV aggregate is recorded but is not interpreted as a performance result; the standalone CSV values are internally consistent and finite.

## 11. Supported Conclusion

“NAFNet-small候选正式配置已完成100-step资源探测和完整validation链路验证，可依据测得的显存、吞吐量和训练成本决定是否进入正式多随机种子训练。”

## 12. Unsupported Conclusions

- 尚未完成正式baseline训练；
- 尚未获得正式baseline validation结果；
- 尚未评估final test；
- 尚未验证三随机种子稳定性；
- 尚未证明200 epoch是最优训练长度；
- 尚未实现颜色算子；
- 尚未实现散射算子；
- 尚未验证算子顺序假设；
- 尚无论文主实验结果。

## 13. Final Phase Status

**Final Phase Status: NEEDS_CONFIGURATION_REVISION**

**next phase authorization=false**

Phase B3a stops here. Phase B3b is not authorized.
