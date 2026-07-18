# Phase B1b NAFNet Cloud Runtime Validation Report

## 1. Scope

This phase archives the already-produced Phase B1b cloud runtime validation evidence. No model source, wrapper, test, protocol, decision, or state file was modified, and no test was rerun while preparing this report.

Phase B1b covers only the fixed-upstream ordinary-NAFNet versus UIE3 minimal-import runtime validation and the project-specific NAFNet-small engineering checks. It does not cover datasets, training, underwater image enhancement performance, restoration operators, operator ordering, or paper experiments.

## 2. Evidence Sources

This report uses only the following supplied records:

- `ORDER_STUDY_PROTOCOL.md`
- `ORDER_STUDY_DECISIONS.md`
- `ORDER_STUDY_STATE.yaml`
- `NAFNET_IMPORT_AUDIT.md`
- `reports/PHASE_B1A_REPORT.md`
- `reports/runtime/phase_b1b_environment.txt`
- `reports/runtime/phase_b1b_git_state.txt`
- `reports/runtime/phase_b1b_test.log`
- `reports/runtime/phase_b1b_exit_code.txt`

No result absent from these records is inferred.

## 3. Recorded Git State

- Tested UIE3 commit: `5f45db37c7156b40c3caff0c4b212169dcb7531c`
- Recorded UIE3 status at validation time: `reports/runtime/` was untracked.
- Upstream NAFNet commit: `2b4af71ebe098a92a75910c233a3965a3e93ede4`
- Recorded upstream NAFNet status: clean (the status section contains no changed paths).
- Report-archive ending commit: not present in the supplied evidence; creating this report does not create or update a Git commit.

## 4. Recorded Runtime Environment

| Field | Recorded value |
|---|---|
| Platform | `Linux-5.15.0-139-generic-x86_64-with-glibc2.35` |
| Python | `3.12.13`, Anaconda build, GCC 14.3.0 |
| Python executable | `/root/miniconda3/envs/pax/bin/python` |
| PyTorch | `2.13.0+cu126` |
| PyTorch path | `/root/miniconda3/envs/pax/lib/python3.12/site-packages/torch/__init__.py` |
| pytest | `9.1.1` |
| CUDA visible devices | empty |
| CUDA available | `False` |
| PyTorch CUDA runtime metadata | `12.6` |
| Test device | CPU |
| Test dtype | float32 |

## 5. Test Execution Summary

- pytest result: **8 passed**
- warnings: **64**
- pytest exit code: **0**
- elapsed test time recorded by pytest: **5.12 seconds**

The eight passing pytest cases comprise the three parameterized official/imported equivalence shapes plus the five remaining required test functions.

## 6. Numerical Equivalence Results

Both models used the fixed research configuration:

```yaml
img_channel: 3
width: 32
enc_blk_nums: [2, 2, 2]
middle_blk_num: 4
dec_blk_nums: [2, 2, 2]
```

| Input shape | Maximum absolute error | Mean absolute error | Result |
|---|---:|---:|---|
| `[1,3,256,256]` | 0 | 0 | passed |
| `[2,3,128,192]` | 0 | 0 | passed |
| `[1,3,257,259]` | 0 | 0 | passed |

All three shapes satisfy the declared thresholds of `max_abs_error <= 1e-6` and `mean_abs_error <= 1e-7`.

## 7. Parameter Count and State-Dict Compatibility

- Official implementation parameter count: **2,846,755**
- UIE3 imported implementation parameter count: **2,846,755**
- Ordered state-dict keys, tensor shapes/dtypes, and strict loading compatibility: passed.
- Missing keys: none indicated by the passing strict-load test.
- Unexpected keys: none indicated by the passing strict-load test.

## 8. Validated Runtime Checks

The recorded passing suite verifies:

- official/imported output equivalence;
- output shape preservation;
- non-multiple-size right/bottom padding and output crop;
- forward and backward execution with finite-gradient checks;
- state-dict compatibility;
- the NAFNet-small global residual is not duplicated.

## 9. Warning Record

The 64 warnings are split evenly:

- 32 warnings from the official upstream `arch_util.py`;
- 32 warnings from the UIE3 vendored `nafnet_arch.py`.

Both report the same PyTorch deprecation:

```text
'saved_variables' is deprecated; use 'saved_tensors'
```

The warnings did not prevent the tests from passing and did not produce a numerical difference under the recorded conditions. They are retained as future PyTorch-compatibility technical debt. Resolving them would modify the audited custom LayerNorm backward source and therefore requires a separately authorized change plus renewed equivalence validation.

## 10. Acceptance Criteria

| Criterion | Evidence | Status |
|---|---|---|
| Fixed upstream commit | `2b4af71ebe098a92a75910c233a3965a3e93ede4` | met |
| Upstream clean | empty recorded NAFNet status | met |
| All required runtime tests pass | `8 passed` | met |
| pytest success | exit code 0 | met |
| Three required shapes tested | all present in raw log | met |
| Numerical error thresholds | max and mean errors are 0 for all shapes | met |
| Equal parameter counts | both are 2,846,755 | met |
| Forward/backward and finite gradients | required test passed | met |
| State-dict compatibility | required test passed | met |
| Padding/cropping | required test passed | met |
| No duplicated global residual | required test passed | met |

## 11. Test Command Record

The literal cloud invocation command is not included in the supplied runtime artifacts, so this report does not invent one. The B1a report records the intended validation command as:

```bash
NAFNET_UPSTREAM_ROOT=../NAFNet PYTHONPATH=. python -m pytest -q -s tests/test_nafnet_import.py
```

The test log and exit-code record are the authoritative execution evidence for this archive.

## 12. Files Added or Modified During Archival

- `reports/PHASE_B1B_REPORT.md`

No other file was created or modified by this archival step.

## 13. Unresolved Issues

- The `ctx.saved_variables` deprecation remains technical debt in both the fixed official source and the identical vendored calculation path.
- The literal cloud shell invocation and separate wall-clock metadata outside pytest's recorded 5.12 seconds are not present in the supplied artifacts.
- Dataset, training, evaluation, and underwater restoration questions remain outside Phase B1b.

## 14. Supported Conclusion

“UIE3中的最小NAFNet迁移实现与固定上游提交，在记录的CPU float32环境和测试条件下计算等价；NAFNet-small研究配置通过前向、反向、尺寸、state_dict、padding/cropping和全局残差验证。”

## 15. Unsupported Conclusions

- 尚未验证LSUI数据读取；
- 尚未建立训练与评价链路；
- 尚未验证UIE收敛和性能；
- 尚未实现颜色算子；
- 尚未实现散射算子；
- 尚未验证顺序假设；
- 尚无论文实验结果。

## 16. Next-Phase Authorization

`current_phase.next_phase_authorized == false`. This report does not update phase state or authorize any later phase. Human acceptance is required before further work.

## 17. Final Phase Status

**PASS**
