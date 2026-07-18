# Phase B1a NAFNet Minimal Import Static Implementation Report

## 1. Scope

Phase B1a created the statically auditable minimal ordinary-NAFNet import, the project-specific NAFNet-small wrapper, and the future Phase B1b runtime test suite. It performed only syntax, AST, dependency, text, provenance, license, and Git checks that do not import or execute PyTorch.

The local Mac Python environment is known not to contain PyTorch. Decision D-0008 explicitly accepts that fact and separates cloud runtime validation into Phase B1b. No dependency was installed, no runtime model test was executed, and Phase B1b was not entered or authorized by this work.

## 2. Starting Git State

- Workspace: `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE3_workspace`
- UIE3 branch: `main`
- UIE3 starting commit: `459f6ca61752289e71ca2bf6aaa4bf54cafc8d1b`
- UIE3 starting worktree: clean (`main` was four commits ahead of `origin/main`)
- Upstream branch: `main`
- Upstream commit: `2b4af71ebe098a92a75910c233a3965a3e93ede4`
- Upstream starting worktree: clean
- Protocol/state version: `1.0.1`
- Authorized phase: `B1a / nafnet_minimal_import_static_implementation`
- Ending commit: unchanged at the starting commit because Phase B1a changes are uncommitted and this task did not authorize creating a Git commit.

## 3. Files Added or Modified

Only phase-authorized files were added:

- `third_party/nafnet/nafnet_arch.py`
- `third_party/nafnet/LICENSE`
- `third_party/nafnet/UPSTREAM.md`
- `src/models/backbones/nafnet_small.py`
- `tests/test_nafnet_import.py`
- `reports/PHASE_B1A_REPORT.md`

No upstream file, governance file, audit, README, dataset, training framework, metric, loss, checkpoint system, operator, order model, or router was modified or created.

## 4. Imported Symbols

Exactly five audited architecture symbols were imported:

1. `LayerNormFunction`
2. `LayerNorm2d`
3. `SimpleGate`
4. `NAFBlock`
5. `NAFNet`

The first two came from upstream `basicsr/models/archs/arch_util.py`; the last three came from upstream `basicsr/models/archs/NAFNet_arch.py`. Static AST dumps of each complete class definition match the pinned upstream definition exactly when line/location attributes are excluded.

## 5. Removed Dependencies

The vendored architecture imports only `torch`, `torch.nn`, and `torch.nn.functional`. The extraction removed:

- the BasicSR import path for `LayerNorm2d` by co-locating the exact LayerNorm implementation;
- the `Local_Base` import;
- `NAFNetLocal` and all Local conversion code;
- `AvgPool2d` from `local_arch.py`;
- BasicSR dynamic discovery/registration;
- BasicSR training, data, loss, metric, logger, and checkpoint infrastructure;
- official YAML files and pretrained weights.

No mathematical implementation was refactored, optimized, substituted, or simplified.

## 6. License and Attribution

- `third_party/nafnet/LICENSE` is byte-for-byte identical to `../NAFNet/LICENSE`.
- Both files have SHA-256 `a29ecef3456149898f08e4c71b11b33e7d333664e087bc212e84e18ddd6599ad` and size 12,466 bytes.
- The vendored source retains the 2022 megvii-model copyright and paper citation.
- It adds the applicable `Copyright 2018-2020 BasicSR Authors` attribution for the extracted LayerNorm source.
- The header records the upstream repository, pinned commit, extraction/consolidation date, removed import coupling, and pending runtime validation.
- `third_party/nafnet/UPSTREAM.md` records the repository, commit, acquisition date, source and destination files, imported symbols, source hashes, removed dependencies, import/consolidation changes, computation-semantics intent, MIT attribution, BasicSR Apache-2.0 attribution, and the B1b command.

## 7. NAFNet-small Configuration

`src/models/backbones/nafnet_small.py` directly subclasses/instantiates the vendored ordinary `NAFNet` with the fixed research configuration:

```yaml
img_channel: 3
width: 32
enc_blk_nums: [2, 2, 2]
middle_blk_num: 4
dec_blk_nums: [2, 2, 2]
padder_size: 8
```

`NAFNetSmall` does not override `forward`; it therefore introduces no wrapper-level residual, clamp, sigmoid, normalization, or pretrained-weight load. Both the class and `build_nafnet_small()` validate all configuration values and raise explicit `ValueError` exceptions for noncanonical inputs.

## 8. Static Checks Performed

The following check categories were actually executed without importing torch:

1. In-memory Python `compile(source, filename, "exec")` for all three new Python files.
2. Python AST parsing of all three new Python files.
3. Per-class AST comparison between the five vendored symbols and their pinned upstream definitions.
4. AST inspection of vendored imports.
5. AST inspection that `NAFNetSmall` has no `forward` override.
6. AST/literal inspection of all fixed wrapper constants.
7. AST enumeration of the six required test functions.
8. Text search for forbidden `NAFNetLocal`, `Local_Base`, and BasicSR imports.
9. Text search for the pinned commit and license attributions in `UPSTREAM.md`.
10. `cmp -s`, SHA-256, line-count, and byte-count comparison of upstream and vendored licenses.
11. Git status/diff checks for authorized scope and governance-file preservation.

No `pytest`, model import, model construction, forward, backward, parameter count, state-dict load, padding execution, or numerical comparison was run.

## 9. Static Check Results

| Check | Result |
|---|---|
| Python syntax compilation | `COMPILE_OK` for vendored architecture, wrapper, and test file |
| Required vendored class set | exactly the five audited classes |
| Class AST equality | `AST_MATCH` for all five symbols |
| Vendored imports | exactly `torch`, `torch.nn`, `torch.nn.functional` |
| `NAFNetLocal` / `Local_Base` | absent |
| BasicSR runtime import | absent |
| Wrapper forward override | absent |
| Fixed wrapper configuration | exact values and padding multiple 8 |
| Six required test names | all present |
| License comparison | byte-identical, same SHA-256 and size |
| UPSTREAM pinned commit | present and exact |

The initial license copy check correctly detected one extra terminal newline in the destination. It was removed, after which the byte comparison and SHA-256 comparison passed. No tolerance or requirement was weakened.

## 10. Runtime Tests Created

`tests/test_nafnet_import.py` contains:

1. `test_official_and_imported_output_equivalence`
2. `test_nafnet_small_output_shape`
3. `test_nafnet_small_non_multiple_size`
4. `test_nafnet_small_forward_backward`
5. `test_nafnet_small_state_dict_compatibility`
6. `test_nafnet_small_global_residual_not_duplicated`

The suite supports `NAFNET_UPSTREAM_ROOT` and otherwise resolves the workspace sibling `../NAFNet`. It verifies the upstream Git commit and loads the official source under `_uie3_official_nafnet_reference`, with stubs only for unused BasicSR/Local framework imports so the official ordinary class cannot alias the vendored class.

The planned runtime conditions are CPU float32, seed 3407, deterministic algorithms, TF32 disabled, eval mode, strict shared state-dict loading, no AMP, no `torch.compile`, and `torch.no_grad()` for output comparisons. The required shapes are `[1,3,256,256]`, `[2,3,128,192]`, and `[1,3,257,259]`; planned thresholds are `max_abs_error <= 1e-6` and `mean_abs_error <= 1e-7`.

The suite also checks total/trainable parameter counts, ordered state keys, tensor shapes/dtypes, missing/unexpected keys, padder multiple 8, right/bottom zero padding, output crop, finite scalar loss, finite input/parameter gradients, official/imported gradient proximity, and the zero-ending-convolution residual guard.

## 11. Runtime Tests Not Executed

All runtime tests remain unexecuted by design. The local Python 3.9.6 environment lacks PyTorch, and Phase B1a forbids installing dependencies or executing PyTorch tests. Consequently, this report contains no runtime parameter count, output error, gradient result, padding/crop result, state-dict load result, or residual-test result.

## 12. Cloud Validation Commands

After separate human authorization of Phase B1b, run from the UIE3 repository root in the cloud PyTorch environment:

```bash
NAFNET_UPSTREAM_ROOT=../NAFNet PYTHONPATH=. python -m pytest -q -s tests/test_nafnet_import.py
```

Before accepting results, Phase B1b must record the Python, PyTorch, pytest, OS, CPU/device, dtype, seed, deterministic settings, upstream commit, complete test output, parameter count, per-shape max/mean errors, and wall time. The command above is a future recommendation only and was not executed in B1a.

## 13. Unresolved Issues

- Runtime numerical equivalence is unverified.
- Forward and backward execution/finite-gradient behavior is unverified.
- State-dict compatibility is statically expected but not runtime-confirmed.
- Padding, cropping, and global-residual behavior are source/AST-matched but not runtime-confirmed.
- The actual total/trainable parameter count is unknown until B1b.
- The cloud Python/PyTorch/pytest versions and deterministic CPU operator path must be recorded during B1b.
- Dataset, image range, metrics, loss, checkpoint behavior, training entry points, and all experimental controls remain outside B1a and unresolved for later phases.

## 14. Supported Conclusions

Only these conclusions are supported:

- The minimal five-symbol vendored source has been established.
- The fixed research-configuration wrapper and future runtime tests have been established.
- Python syntax compilation and the specified static structure, dependency, provenance, and license checks passed.

## 15. Unsupported Conclusions

Phase B1a does not support claims that:

- the official and imported implementations are numerically equivalent;
- forward/backward behavior is correct;
- state dictionaries are completely runtime-compatible;
- arbitrary-size padding/cropping or the residual guard passes at runtime;
- NAFNet-small improves underwater images or trains/converges;
- color/scattering operators or order hypotheses are valid;
- any result is suitable for a paper conclusion.

## 16. Final Phase Status

**IMPLEMENTED_NOT_RUNTIME_VALIDATED**

The authorized static implementation and static checks are complete. Human review is required. Phase B1b is not authorized and was not started.

本阶段只完成了NAFNet最小迁移的静态实现。由于本地环境没有PyTorch，尚未运行官方与迁移实现的一致性、forward、backward、padding、裁剪和全局残差测试。任何运行时正确性和科研结论都尚未得到验证。
