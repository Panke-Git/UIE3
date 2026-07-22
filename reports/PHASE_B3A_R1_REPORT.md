# Phase B3a-r1 Report — Formal Hardware and Training Budget Freeze

## 1. Phase Objective and Governance

- Phase: `B3a-r1`
- Phase name: `formal_hardware_and_training_budget_freeze`
- Objective: archive the human-approved formal-training hardware policy, freeze the NAFNet-small baseline resource and optimization settings, and define the maximum training budget without running another experiment.
- Current phase authorized: `true`
- Next phase authorized: `false`
- Starting commit: `33901fc9c636d7995d62420a01ea9f2097a65443`
- Ending commit: `pending` (this report is not committed by this archival task)
- Changed file: `reports/PHASE_B3A_R1_REPORT.md`
- Assumption: Decision D-0021 is the accepted, phase-specific authority for the formal-training platform. It distinguishes that platform from the RTX 3060 12 GB development and lightweight-smoke role.

No training, evaluation, testing, configuration edit, or runtime-evidence regeneration was performed in Phase B3a-r1. Existing source, tests, configuration, manifests, governance files, and Phase B3a evidence remain unchanged.

## 2. Evidence and Commands

This report is based on:

- `ORDER_STUDY_PROTOCOL.md`
- `ORDER_STUDY_DECISIONS.md`, especially accepted Decision D-0021
- `ORDER_STUDY_STATE.yaml`
- `configs/nafnet_small_lsui_formal.yaml`
- `reports/PHASE_B3A_REPORT.md`
- `reports/runtime/phase_b3a/environment.txt`
- `reports/runtime/phase_b3a/resource_probe.log`
- `reports/runtime/phase_b3a/resource_probe.json`
- `reports/runtime/phase_b3a/full_validation.log`
- `reports/runtime/phase_b3a/full_validation_metrics.csv`

Commands executed in Phase B3a-r1 were limited to read-only file and Git inspection. No experiment command was run. The historical Phase B3a training and validation commands were not rerun.

## 3. Why Phase B3a Required Configuration Revision

The Phase B3a runtime chain itself passed all recorded acceptance checks:

- the canonical formal train and validation manifests loaded successfully;
- 100 CUDA optimizer steps completed and reached `global_step=100`;
- the recorded train loss was finite;
- no CUDA OOM or data-loader worker error was recorded;
- the checkpoint was saved with both model and optimizer states;
- the complete 385-sample validation pipeline exited successfully;
- every per-image RGB PSNR and SSIM value was finite;
- the final test split was not loaded or evaluated.

The only blocking item was the previous 12 GB portability assumption. With `batch_size=4`, `patch_size=256`, and AMP enabled, the RTX 3090 probe recorded a sampled peak GPU memory use of `17148.0` MiB. That value exceeds the former requirement that the complete formal configuration fit an RTX 3060 12 GB target. Therefore Phase B3a reported `NEEDS_CONFIGURATION_REVISION`; it did not report a runtime or scientific failure on the actual RTX 3090 cloud platform.

## 4. Frozen Formal Hardware Policy

- **Formal training platform:** NVIDIA RTX 3090 24 GB, or an equivalent GPU with sufficient memory for the frozen configuration.
- **Development platform:** RTX 3060 12 GB is limited to local development and lightweight smoke testing.
- **Portability requirement:** the complete formal training configuration is no longer required to fit within 12 GB.
- **Resource basis:** the frozen formal platform is the class of hardware on which the recorded 100-step probe completed without CUDA OOM at `batch_size=4`, with a sampled peak of `17148.0` MiB.

This role separation removes the batch-size hardware ambiguity: formal training targets a 24 GB-class platform, while 12 GB hardware is not a formal-run acceptance target.

## 5. Frozen Formal Resource and Optimization Configuration

| Setting | Frozen value | Status |
| --- | --- | --- |
| Batch size | `4` | **FREEZE** |
| Patch size | `256` | **FREEZE** |
| Data-loader workers | `4` | **FREEZE** |
| AMP | `true` | **FREEZE** |
| Optimizer | `AdamW` | **FREEZE** |
| Learning rate | `2.0e-4` | **FREEZE** |
| Weight decay | `0` | **FREEZE** |

These values already match `configs/nafnet_small_lsui_formal.yaml`; no configuration, model, data, or training-source change is required. The freeze is a protocol decision based on the accepted formal platform and resource probe. It is not a claim that the optimizer or learning rate was performance-optimized.

## 6. Frozen Training Budget and Selection Rules

- Maximum training length: `200` epochs — **FREEZE_AS_MAXIMUM_BUDGET**.
- The 200-epoch limit is a maximum execution budget, not evidence that 200 epochs is performance-optimal.
- Checkpoint selection rule: select the checkpoint with the highest formal-validation RGB PSNR.
- The held-out final test split must not participate in checkpoint selection, hyperparameter selection, model selection, or training feedback.

Formal random seeds are frozen as:

1. `3407`
2. `1234`
3. `2027`

Formal results must retain per-seed outputs and follow the protocol's multi-seed reporting requirements; reporting only the best seed is prohibited.

## 7. Recorded Resource Cost

The following measured estimates come from `resource_probe.json`:

| Field | Recorded value |
| --- | ---: |
| Probe optimizer steps | `100` |
| Probe wall time | `60.162727` seconds |
| Estimated training-only duration | `39.429545` seconds |
| Estimated seconds per training step | `0.39429545` |
| Estimated training images per second | `10.144677043572276` |
| Estimated steps per epoch | `867` |
| Estimated training-only epoch time | `341.85415515` seconds |
| Estimated total epoch time | `362.58733715` seconds |
| Estimated total epoch time | `6.043122285833333` minutes |
| Estimated 200-epoch time per seed | `20.14374095277778` hours, approximately `20.14` hours |
| Derived three-seed estimate | `60.43122285833334` hours, approximately `60.43` hours |
| Full-validation elapsed time | `20.733182` seconds |
| Peak sampled GPU memory | `17148.0` MiB |

The three-seed figure is the arithmetic product of the recorded per-seed 200-epoch estimate and three formal seeds. Both figures are planning estimates based on a 100-step RTX 3090 probe; actual time may differ. Startup and checkpoint overhead remain included, GPU telemetry used one-second sampling and may miss short peaks, long-run stability has not been established, and scheduling or other operational overhead is excluded.

## 8. Acceptance Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Formal execution platform is explicitly defined | **PASS** | RTX 3090 24 GB or an equivalent sufficient-memory GPU is the formal platform. |
| Batch size has no remaining hardware-target ambiguity | **PASS** | `batch_size=4` is frozen for the 24 GB-class formal platform; 12 GB is limited to development and smoke use. |
| All formal resource parameters are explicitly frozen | **PASS** | Batch size, patch size, workers, AMP, optimizer, learning rate, and weight decay are all marked `FREEZE`. |
| 200 epochs is a maximum budget rather than an optimality claim | **PASS** | It is explicitly marked `FREEZE_AS_MAXIMUM_BUDGET`; no performance-optimality claim is made. |
| Checkpoint selection rule is explicit | **PASS** | Highest formal-validation RGB PSNR. |
| Final test isolation rule is explicit | **PASS** | Final test is excluded from checkpoint, hyperparameter, and model selection and from training feedback. |
| Model, data, configuration, and training source require no modification | **PASS** | The existing formal configuration already contains the frozen values; this phase is report-only. |
| B3a resource probe does not need to be rerun | **PASS** | The human-approved policy adopts the measured RTX 3090-class platform, and no new runtime question is introduced by this archival freeze. |

All Phase B3a-r1 acceptance criteria are satisfied.

## 9. Remaining Limitations and Risks

- 尚未完成正式baseline训练；
- 尚未获得三seed正式结果；
- 尚未评估final test；
- 尚未证明200 epochs性能最优；
- 尚未实现颜色算子；
- 尚未实现散射算子；
- 尚未验证算子顺序假设。
- The approximately 20.14-hour per-seed and 60.43-hour three-seed costs are estimates, not observed full-run durations.
- A nominally equivalent GPU must still provide sufficient available memory and compatible CUDA/PyTorch behavior; “equivalent” is not a performance guarantee.

## 10. Supported Conclusion

“NAFNet-small正式baseline的执行硬件、资源配置、优化协议、最大训练预算、随机种子和checkpoint选择规则已经冻结，可以在单独人工授权后进入三随机种子正式训练。”

## 11. Unsupported Conclusions

- 尚未完成正式baseline训练；
- 尚未获得三seed正式结果；
- 尚未评估final test；
- 尚未证明200 epochs性能最优；
- 尚未实现颜色算子；
- 尚未实现散射算子；
- 尚未验证算子顺序假设。

No formal baseline performance, convergence, final-test, operator, or order-study conclusion is supported by Phase B3a-r1.

## 12. Final Phase Status

**Final Phase Status: PASS**

**next phase authorization=false**

Phase B3a-r1 stops here. Phase B3b remains unauthorized pending separate human authorization.
