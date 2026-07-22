# Phase B3b-1r1 Report — AMP Overflow Handling Remediation

## 1. Phase Objective and Governance

- Phase: `B3b-1r1`
- Phase name: `amp_overflow_handling_remediation`
- Objective: make CUDA AMP overflow a recoverable skipped optimizer update, prevent skipped updates from incrementing `global_step`, and retain strict non-AMP failure semantics.
- Current phase authorized: `true`
- Next phase authorized: `false`
- Starting commit: `d6652a2b08c323d7fb731d40faa6e6ca56a3f8cc`
- Ending commit: `pending`
- Files changed by this remediation:
  - `src/engine/trainer.py`
  - `tests/test_train_step.py`
  - `reports/PHASE_B3B_1R1_REPORT.md`

The worktree already contained user changes to `ORDER_STUDY_DECISIONS.md` and `ORDER_STUDY_STATE.yaml` before implementation. Those governance changes were read and preserved but not modified by this remediation.

## 2. Evidence Availability

- Accepted Decision D-0023 records that formal seed 3407 completed four validation epochs, reached a last completed `global_step` of 3468, and then stopped on an unscaled non-finite parameter gradient while preceding train loss and validation metrics were finite.
- `reports/runtime/phase_b3b_seed3407/train.log` is not present in the current local worktree, so this report does not claim to quote or independently reparse that log.
- `reports/PHASE_B3B_SEED3407_REPORT.md` is also not present.
- The failure path is directly present in the pre-remediation `BaselineTrainer.train_step`: `GradScaler.unscale_()` was immediately followed by a fatal finite-gradient assertion before `GradScaler.step()` and `GradScaler.update()`.

## 3. Original Failure Mechanism

The old AMP path executed:

```text
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
fatal finite-gradient assertion
scaler.step(optimizer)
scaler.update()
global_step += 1
```

When unscaled gradients contained NaN or Inf, the assertion raised `FloatingPointError` before GradScaler could use its already recorded `found_inf` state. This converted an expected recoverable AMP overflow into a fatal training failure. Separately, the unconditional `global_step += 1` encoded attempted batches rather than successfully applied optimizer updates.

## 4. Remediated AMP Semantics

The required operation order remains unchanged:

```text
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
optional clipping for finite gradients only
scaler.step(optimizer)
scaler.update()
```

After `unscale_()`, gradients are inspected through public tensor operations:

- If AMP is enabled and any gradient is non-finite, clipping is skipped and no `FloatingPointError` is raised.
- `scaler.step(optimizer)` is still called so GradScaler can skip the optimizer update from its recorded overflow state.
- `scaler.update()` is still called so the public scale is reduced.
- `amp_scale_before` is recorded with public `GradScaler.get_scale()` before step/update.
- `amp_scale_after` is recorded with the same public API after update.
- `amp_scale_after < amp_scale_before` defines an AMP overflow and a skipped optimizer update.
- The return metadata contains `optimizer_step_applied`, `amp_overflow_detected`, `amp_scale_before`, and `amp_scale_after`.
- A structured `AMP_OVERFLOW_DETECTED` line is emitted to stdout with both scales and the unchanged `global_step`, making an overflow visible in the formal training log.
- No private `torch.amp` fields or APIs are accessed, and the return value of `scaler.step()` is not used to infer whether an update occurred.

For a finite AMP step, clipping remains after unscale, the optimizer is updated once, the scale may remain unchanged or grow, `optimizer_step_applied=true`, and `amp_overflow_detected=false`.

## 5. Non-AMP and Gradient-Clipping Semantics

- When AMP is disabled, non-finite gradients still call the existing strict assertion and raise `FloatingPointError` before any optimizer update.
- Gradient clipping runs only when the unscaled gradients are finite.
- Finite-gradient clipping continues to use `clip_grad_norm_(..., error_if_nonfinite=True)` followed by the existing finite-gradient assertion.
- AMP non-finite gradients bypass clipping and are delegated to GradScaler.
- No clipping setting was added to the formal configuration; the frozen formal value remains `gradient_clip_norm: null`.

## 6. `global_step` Definition

`global_step` now means the number of optimizer updates that were actually applied successfully:

- successful finite optimizer update: increment by one;
- AMP overflow and skipped optimizer update: do not increment;
- non-AMP non-finite gradient: raise before any increment.

This definition is also the value persisted and restored by the unchanged checkpoint format.

## 7. Tests Added

`tests/test_train_step.py` now covers:

1. A finite CUDA AMP step updates parameters, increments `global_step`, and returns successful AMP metadata.
2. A controlled Inf gradient under CUDA AMP does not raise, does not update parameters, does not increment `global_step`, reports overflow, reduces the scale, and emits an overflow log record.
3. A finite step after an overflow continues normally and increments `global_step` once.
4. A controlled non-AMP Inf gradient still raises `FloatingPointError` without changing parameters or `global_step`.
5. The finite gradient-clipping path retains bounded gradients, applies one optimizer update, and increments `global_step` once.
6. AMP scaler state and the successful-update `global_step` survive checkpoint save/resume; a skipped overflow remains excluded and the next finite update continues strictly.

The existing CPU train-step test was also strengthened to check the new return metadata. Existing unrelated tests were not removed or relaxed. No test uses LSUI data.

## 8. Test and Static-Check Results

| Check | Result |
| --- | --- |
| `src/engine/trainer.py` AST parse | `PASS` |
| `src/engine/trainer.py` compile from source | `PASS` |
| `tests/test_train_step.py` AST parse | `PASS` |
| `tests/test_train_step.py` compile from source | `PASS` |
| `git diff --check` before report creation | `PASS` |
| Local pytest | `NOT_RUN` — local Python has neither torch nor pytest |
| CUDA AMP tests | `NOT_RUN` — cloud CUDA validation is pending |

No dependency was installed. The runtime assertions in the newly added tests must be executed in the authorized cloud PyTorch/CUDA environment before seed 3407 is resumed.

## 9. Checkpoint Compatibility

- `src/engine/checkpoint.py` was not modified.
- The checkpoint schema and fields are unchanged.
- Scaler state remains saved through `scaler_state_dict` and restored through the existing strict resume path.
- `global_step` remains stored in the same field; only its runtime meaning is corrected to count successful optimizer updates.

## 10. Formal Configuration and Run State

- No file under `configs/**` was modified.
- Formal manifests, model structure, loss, optimizer, learning rate, weight decay, batch size, patch size, AMP setting, seed, and maximum epoch budget remain unchanged.
- The preserved seed-3407 checkpoints and failure evidence were not modified.
- Formal seed 3407 training has **not** been resumed in this phase.
- No training or evaluation command was run.

## 11. Acceptance Status and Remaining Risks

| Requirement | Static status |
| --- | --- |
| Preserve backward → unscale → optional clip → step → update order | `IMPLEMENTED` |
| AMP non-finite gradients do not become an immediate fatal assertion | `IMPLEMENTED` |
| AMP skipped update does not increment `global_step` | `IMPLEMENTED` |
| Finite AMP update increments `global_step` exactly once | `IMPLEMENTED` |
| Non-AMP non-finite gradient remains fatal | `IMPLEMENTED` |
| Finite clipping behavior remains strict | `IMPLEMENTED` |
| Public scale comparison determines AMP overflow | `IMPLEMENTED` |
| Checkpoint format remains compatible | `IMPLEMENTED` |
| Cloud CUDA runtime validation | `PENDING` |

Remaining risks:

- The controlled CUDA overflow, scale reduction, parameter immutability, recovery step, and scaler resume tests have not yet run in the cloud.
- The missing local `train.log` prevents independent archival verification of the exact failure line in this task; D-0023 is the accepted record.
- Seed 3407 must not resume until the remediation suite passes in the intended cloud CUDA environment.

## 12. Supported and Unsupported Conclusions

Supported:

- The trainer and regression tests statically implement the required AMP overflow and successful-update `global_step` semantics without changing the checkpoint schema or formal scientific configuration.

Unsupported:

- The remediation has not yet been validated by cloud CUDA runtime tests.
- Formal seed 3407 training has not resumed or completed.
- No formal validation or final-test result is produced by this phase.
- No model-performance, convergence, operator, order, or paper conclusion is supported.

## 13. Final Phase Status

**Final Phase Status: IMPLEMENTED_NOT_CLOUD_RUNTIME_VALIDATED**

**next phase authorization=false**

Phase B3b-1r1 stops here. Formal seed 3407 remains paused pending separate cloud runtime validation and human authorization.
