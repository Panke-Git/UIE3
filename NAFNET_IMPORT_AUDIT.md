# NAFNet Minimal Import Audit

- Audit date: 2026-07-18 (Asia/Shanghai)
- Authorized phase: Phase A
- Authorized task: `nafnet_minimal_import_audit`
- Scope: read-only inspection of `../NAFNet` and `.`; this document is the only authorized write.
- Important boundary: this audit does not authorize source import, implementation, dependency installation, weight download, training, or Phase B.

## 1. Workspace Validation

The required control documents were read completely and in the required order before any workspace or repository check:

1. `UIE3/ORDER_STUDY_PROTOCOL.md`
2. `UIE3/ORDER_STUDY_DECISIONS.md`
3. `UIE3/ORDER_STUDY_STATE.yaml`

The control documents and the Phase A prompt are consistent. The state file confirms all of the following:

- `current_phase.id == A`
- `current_phase.name == nafnet_minimal_import_audit`
- `current_phase.authorized == true`
- `current_phase.next_phase_authorized == false`
- the only writable path is `UIE3/NAFNET_IMPORT_AUDIT.md`
- modifying the upstream repository, copying source, implementing any model/operator, installing dependencies, downloading weights, training, and automatic phase advancement are forbidden.

The initial checks produced:

| Check | Result |
|---|---|
| `pwd` | `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE3_workspace` |
| `./NAFNet` | exists and is a Git repository |
| `./UIE3` | exists and is a Git repository |
| expected workspace name | `UIE3_workspace` confirmed |

No path was guessed or substituted. `NAFNet` was treated as read-only. The three control files were not changed, and this audit does not update phase state or authorize Phase B.

## 2. Git Status and Upstream Version

### Repository state at audit start

| Repository | Branch | Commit SHA | Status | Remote |
|---|---|---|---|---|
| `NAFNet` | `main` | `2b4af71ebe098a92a75910c233a3965a3e93ede4` | clean; no tracked, staged, or untracked changes | `https://github.com/megvii-research/NAFNet.git` |
| `UIE3` | `main` | `1751802240fe962402de5386d1fef1957f0fcada` | not clean; four untracked files existed before this audit | `https://github.com/Panke-Git/UIE3.git` |

The pre-existing untracked UIE3 files are:

- `.DS_Store`
- `ORDER_STUDY_DECISIONS.md`
- `ORDER_STUDY_PROTOCOL.md`
- `ORDER_STUDY_STATE.yaml`

They were read where authorized and were not overwritten or modified. `NAFNet` remained unmodified during the audit.

### Upstream version and license

- Upstream HEAD: `2b4af71ebe098a92a75910c233a3965a3e93ede4`
- Branch: `main`
- Commit subject: `Merge pull request #96 from OliverGrace/patch-1`
- Commit date: 2024-03-29T00:52:42+08:00
- Root license file: `NAFNet/LICENSE`
- Primary NAFNet copyright notice: `Copyright (c) 2022 megvii-model`
- Root license for NAFNet-authored material: MIT License
- Bundled BasicSR notice: `Copyright 2018-2020 BasicSR Authors`
- Bundled BasicSR license: Apache License 2.0

The repository explicitly says the implementation is based on BasicSR, and several files carry “Modified from BasicSR” headers. The root `LICENSE` contains both the NAFNet MIT text and the BasicSR Apache-2.0 text. Both license families and the applicable copyright/attribution notices must be preserved for imported portions.

## 3. Official NAFNet Core Implementation

All paths in this section are relative to the upstream `NAFNet` repository.

| Path | Symbol or mechanism | Needed for ordinary `NAFNet`? | BasicSR-only? | NAFNetLocal-only? | May be omitted from minimal import? |
|---|---|---:|---:|---:|---:|
| `basicsr/models/archs/NAFNet_arch.py` | `NAFNet` (lines 83-162) | yes | no | no | no |
| `basicsr/models/archs/NAFNet_arch.py` | `NAFBlock` (lines 27-80) | yes | no | no | no |
| `basicsr/models/archs/NAFNet_arch.py` | `SimpleGate` (lines 22-25) | yes | no | no | no |
| `basicsr/models/archs/arch_util.py` | `LayerNorm2d` (lines 291-300) | yes | no, but located in a BasicSR-derived utility file | no | no |
| `basicsr/models/archs/arch_util.py` | `LayerNormFunction` (lines 264-289) | yes, including backward semantics | no, but located in a BasicSR-derived utility file | no | no |
| `basicsr/models/archs/NAFNet_arch.py` | `NAFNetLocal` (lines 164-174) | no | no | yes | yes |
| `basicsr/models/archs/local_arch.py` | `Local_Base` (lines 99-104) | no | no | yes | yes |
| `basicsr/models/archs/local_arch.py` | `AvgPool2d`, `replace_layers` | no | no | yes | yes |
| `basicsr/models/archs/__init__.py` | `_arch.py` scan, `dynamic_instantiation`, `define_network` | no | yes | no | yes |
| `basicsr/utils/misc.py` | `scandir` used by dynamic discovery | no | yes | no | yes |
| `basicsr/models/image_restoration_model.py` | calls `define_network` | no | yes | no | yes |

This upstream commit does **not** use an `ARCH_REGISTRY` decorator or a standalone registry module for NAFNet. Registration is convention-based dynamic discovery: `basicsr/models/archs/__init__.py` scans files ending in `_arch.py`, imports them, and resolves the YAML `type` string by class name. That discovery path is part of BasicSR integration, not ordinary `NAFNet.forward`, and must not be imported into UIE3 merely to construct NAFNet.

The only external runtime library required by the minimal ordinary architecture is PyTorch. Relevant primitives are `torch.autograd.Function`, `torch.nn.Module`, convolution, adaptive average pooling, pixel shuffle, sequential/module lists, parameter tensors, identity/dropout, and `torch.nn.functional.pad`. NumPy is used by `local_arch.py` but is not a dependency of ordinary NAFNet once the Local variant is excluded.

## 4. Official Dependency Graph

The complete direct dependency chain for ordinary NAFNet is:

```text
NAFNet
├── PyTorch modules
│   ├── Conv2d (intro, ending, downsampling, up-projection)
│   ├── PixelShuffle
│   ├── ModuleList / Sequential
│   └── functional.pad
└── NAFBlock
    ├── Conv2d (pointwise and depthwise)
    ├── AdaptiveAvgPool2d (simplified channel attention)
    ├── Dropout or Identity
    ├── learned beta and gamma parameters
    ├── SimpleGate
    └── LayerNorm2d
        └── LayerNormFunction
            └── PyTorch custom autograd forward/backward
```

Ordinary NAFNet has no forward dependency on `NAFNetLocal`, `Local_Base`, `AvgPool2d` from `local_arch.py`, the BasicSR dynamic architecture loader, BasicSR models, datasets, losses, metrics, logging, configuration parsing, or training code.

The following semantics cannot be removed or substituted without risking a computation change:

- `LayerNormFunction` normalizes across channel dimension for every spatial position with `eps=1e-6`, and defines a custom backward. Replacing it with a differently shaped `nn.LayerNorm`, batch normalization, group normalization, or another formula is not acceptable for the minimal import.
- `SimpleGate` splits channels into two equal chunks and multiplies them elementwise.
- the depthwise convolution grouping, simplified channel-attention pooling/convolution, `beta`/`gamma` residual scaling, skip additions, down/up operations, and module/parameter names must remain exact.
- `check_image_size`, zero padding, the one whole-network residual addition, and final crop must remain exact.
- keeping module and parameter names is required for direct `state_dict` compatibility.

## 5. Official Configuration and Forward Semantics

### Published width-32 configurations

The constructor default for `img_channel` is 3; the official YAML files omit it and therefore use 3.

| Official config | Type | `img_channel` | `width` | `enc_blk_nums` | `middle_blk_num` | `dec_blk_nums` | training crop / `train_size` |
|---|---|---:|---:|---|---:|---|---|
| `options/train/GoPro/NAFNet-width32.yml` | `NAFNetLocal` | 3 (default) | 32 | `[1, 1, 1, 28]` | 1 | `[1, 1, 1, 1]` | `gt_size: 256`; YAML does not pass `train_size`, so `NAFNetLocal` uses `(1, 3, 256, 256)` |
| `options/train/SIDD/NAFNet-width32.yml` | ordinary `NAFNet` | 3 (default) | 32 | `[2, 2, 4, 8]` | 12 | `[2, 2, 2, 2]` | `gt_size: 256`; ordinary `NAFNet` has no `train_size` argument |

The corresponding GoPro and SIDD test YAML files use the same respective network structures. There is therefore no single official architecture denoted only by “width=32”; the dataset/task configuration also determines depth and, for GoPro, selects the Local variant.

The research configuration required by the study is:

```yaml
img_channel: 3
width: 32
enc_blk_nums: [2, 2, 2]
middle_blk_num: 4
dec_blk_nums: [2, 2, 2]
```

It is not equal to either published width-32 configuration above: it has three encoder/decoder scales rather than four, and different block counts. `NAFNet-small` is a project-specific research configuration of the official `NAFNet` class, **not** an official released fixed model name and **not** an official pretrained model. A later implementation must wrap/configure the unaltered ordinary NAFNet class with these research values; it must not alter an official YAML, mislabel a published checkpoint, or claim official pretraining.

### Forward semantics

- Input: a rank-4 tensor `[B, img_channel, H, W]`; for this study, `[B, 3, H, W]`.
- Output: `[B, img_channel, H, W]`; for this study, exactly `[B, 3, H, W]`.
- Internal size rule: height and width are padded to a multiple of `2 ** len(enc_blk_nums)`.
- Research configuration multiple: 8 because it has three encoder levels.
- Published width-32 GoPro/SIDD multiple: 16 because each has four encoder levels.
- External callers do not need to pre-pad a non-multiple input; `check_image_size` does it internally.
- Padding is appended only on the right and bottom. The upstream call does not specify a mode, so PyTorch `F.pad` uses constant zero padding.
- The network processes the padded tensor, applies the ending convolution, and then performs exactly one global residual addition with the padded input.
- The result is cropped as the top-left `H x W` region, restoring the original spatial shape.
- There is no extra output clamp, sigmoid, normalization, or range conversion in ordinary NAFNet.
- Ordinary `NAFNet` constructs every `NAFBlock` with the default zero dropout rate, so the dropout modules are identities. Its forward result has no special train/eval branch. `NAFNetLocal` calls `eval()` during conversion, but that is excluded from the planned import.

The required whole-network rule is therefore conceptually:

```text
padded_input = zero_pad_right_bottom(input)
output = restoration_branch(padded_input) + padded_input
output = output[:, :, :original_height, :original_width]
```

A UIE3 wrapper must **not** add `output = output + input` again. Doing so would duplicate the official global residual.

## 6. UIE3 Repository Structure

Before this audit document was created, the complete non-Git UIE3 tree was:

```text
UIE3/
├── .DS_Store                         # untracked
├── ORDER_STUDY_DECISIONS.md          # untracked, human-supplied control file
├── ORDER_STUDY_PROTOCOL.md           # untracked, human-supplied control file
├── ORDER_STUDY_STATE.yaml            # untracked, human-supplied control file
└── README.md                          # tracked; contains only the project title
```

There are no dataset, checkpoint, result, cache, or other large directories to omit. The only tracked file at `HEAD` is `README.md`.

The requested integration-point audit found:

| Concern | UIE3 finding |
|---|---|
| train entry | absent |
| test/eval entry | absent |
| dataset | absent |
| dataloader | absent |
| model factory | absent |
| model registry | absent |
| loss | absent |
| metric | absent |
| checkpoint system | absent |
| logger | absent |
| TensorBoard integration | absent |
| config or argparse | absent |
| tests directory | absent |
| third-party code directory | absent |

There is no existing engineering convention to reuse or conflict with. Future training-system design remains outside Phase A and must not be inferred from the upstream BasicSR framework.

## 7. Existing Compatible Components in UIE3

Repository-wide name/content searches (excluding the protocol controls) found none of the following:

- `NAFNet`
- `NAFBlock`
- `LayerNorm2d`
- a U-Net or another restoration backbone
- `third_party/`
- `vendor/`
- a license aggregation file
- a model unit-test framework

Consequently, there is no same-name implementation that a minimal import would overwrite. There is also no existing `NAFNet-small` configuration that conflicts with the protocol configuration. This resolves the configuration rule in Decision D-0003 without selecting between competing UIE3 definitions.

## 8. Minimum Required Import Set

### A. Must import in a later, separately authorized phase

No code was imported in Phase A. The future minimum is the following set of code units, not the complete upstream files:

| Official path | Required symbols/behavior | Why required | Direct dependencies | Import rewrite? | Attribution |
|---|---|---|---|---:|---|
| `basicsr/models/archs/NAFNet_arch.py` | `SimpleGate`, `NAFBlock`, `NAFNet`, including `check_image_size` | defines the complete ordinary architecture and forward path | PyTorch plus `LayerNorm2d` | yes: point `LayerNorm2d` to the vendored minimal module; remove `Local_Base` | retain 2022 megvii-model header and paper citation; distribute MIT text |
| `basicsr/models/archs/arch_util.py` | `LayerNormFunction`, `LayerNorm2d` only | exact channel-wise normalization and custom backward used by every block | PyTorch only after extraction | yes: omit unrelated BasicSR utilities and logger import | retain relevant megvii-model and BasicSR notices; distribute Apache-2.0 text and mark the extracted file as modified |
| `LICENSE` | complete composite license text | supplies the upstream MIT and BasicSR Apache-2.0 terms | none | no semantic rewrite | preserve verbatim |

These five architecture symbols are the minimum executable source set:

1. `LayerNormFunction`
2. `LayerNorm2d`
3. `SimpleGate`
4. `NAFBlock`
5. `NAFNet`

They may be placed together in one clearly attributed vendored `nafnet_arch.py` to avoid importing the rest of `arch_util.py`. That consolidation is an import-path/licensing change only; the class bodies, tensor operations, defaults, module names, parameter names, padding, residual, and crop semantics must remain computation-equivalent.

## 9. Dependencies to Remove or Replace

### B. Lightweight equivalent dependency removal

| Original dependency | Later removal/replacement | Why ordinary forward is unchanged | Required verification |
|---|---|---|---|
| `from basicsr.models.archs.arch_util import LayerNorm2d` | co-locate or use a relative import for the exact `LayerNormFunction`/`LayerNorm2d` implementation | changes only Python import location | strict state keys plus output/gradient equivalence |
| `from basicsr.models.archs.local_arch import Local_Base` | delete the import together with `NAFNetLocal` | neither is referenced by ordinary `NAFNet` or `NAFBlock` | import ordinary model without `local_arch`; output equivalence |
| `_arch.py` scanning, `dynamic_instantiation`, `define_network`, and `scandir` | instantiate the vendored class directly from a small UIE3 wrapper; integrate with a future UIE3 factory only if one is later created | object construction route does not participate in forward math | identical constructor values, keys, parameters, and outputs |
| whole `arch_util.py` imports, including BasicSR logger and unrelated helpers | retain only the two LayerNorm symbols and their PyTorch imports | removed functions are not reachable from ordinary NAFNet | source dependency audit and forward/backward equivalence |
| official task YAML | express the five research constructor arguments in a UIE3 wrapper/config | the official YAMLs are task recipes, not dependencies of the class | assert the wrapper exposes exactly the declared research configuration |

There is no BasicSR registry decorator to remove in this upstream commit. The framework coupling to remove is dynamic module discovery and YAML class-name lookup.

The custom LayerNorm formula, PyTorch primitive choices, convolution attributes, adaptive pooling, pixel shuffle, skip connections, `beta`/`gamma`, zero padding, crop, and single global residual are not framework dependencies and must not be “simplified” away.

## 10. Files That Must Not Be Imported

### C. Excluded upstream material

| Upstream material | Reason not to import |
|---|---|
| `NAFNetLocal` in `NAFNet_arch.py` | Local/TLC inference variant; explicitly out of scope and not needed by ordinary NAFNet |
| `basicsr/models/archs/local_arch.py` (`Local_Base`, `AvgPool2d`, `replace_layers`) | serves Local conversion only; introduces NumPy and alternate pooling behavior |
| remainder of `basicsr/models/archs/arch_util.py` | unrelated residual, upsample, optical-flow, initialization, logging, and benchmarking utilities |
| `basicsr/models/archs/__init__.py` and BasicSR dynamic discovery | framework registration/instantiation, not architecture math |
| `basicsr/models/**` other than the identified symbol sources | complete BasicSR model/training/checkpoint framework is out of scope |
| `basicsr/data/**` | official GoPro/SIDD/REDS and other dataset/dataloader infrastructure is unrelated to UIE3 minimal inference |
| `basicsr/train.py`, `basicsr/test.py`, `basicsr/demo*.py`, `predict.py` | official training, testing, and demo entry points would create a second framework |
| `basicsr/models/losses/**`, `basicsr/metrics/**`, schedulers, loggers, TensorBoard/W&B support | not direct architecture dependencies; UIE3 must define/reuse its own systems later |
| `options/train/**`, `options/test/**` | task-specific reference configurations, useful only as audit evidence |
| `scripts/**`, `docs/**`, `demo/**`, `figures/**` | data preparation, documentation, examples, and media do not execute ordinary NAFNet |
| `Baseline_arch.py`, `NAFSSR_arch.py` | different models, outside the study backbone |
| `setup.py`, `setup.cfg`, `requirements.txt`, `VERSION`, `cog.yaml` | packages and configures the full upstream project; not needed for the five-symbol import |
| pretrained weights or weight-download links/scripts | explicitly forbidden; research `NAFNet-small` has no claimed official checkpoint |

The official repository must remain a read-only reference. The later import must copy only the audited code units and required license/attribution files, not the repository or BasicSR tree wholesale.

## 11. Proposed UIE3 Integration Layout

UIE3 has no existing source layout, so the protocol’s suggested separation is compatible and should be used as the initial proposal:

```text
UIE3/
├── third_party/
│   └── nafnet/
│       ├── nafnet_arch.py       # attributed minimal five-symbol upstream implementation
│       ├── LICENSE              # complete upstream composite license
│       └── UPSTREAM.md          # provenance and modification record
├── src/
│   └── models/
│       └── backbones/
│           └── nafnet_small.py  # thin research-configuration wrapper; no extra residual
└── tests/
    └── test_nafnet_import.py    # planned equivalence/unit tests
```

Package `__init__.py` files may be added only where the eventual UIE3 Python packaging/import strategy requires them. Because no packaging/configuration convention exists yet, Phase A does not prescribe a second training framework or a model registry. The key separation is stable: attributed upstream-derived math in `third_party/nafnet`, project-specific configuration in `src/models/backbones`, and equivalence evidence in `tests`.

## 12. License and Attribution Requirements

The future import must satisfy both sets of obligations represented in upstream `LICENSE`:

1. Preserve the NAFNet MIT copyright notice (`Copyright (c) 2022 megvii-model`) and the MIT permission text in copies or substantial portions.
2. Preserve applicable `Copyright 2018-2020 BasicSR Authors` and source attribution notices for BasicSR-derived material.
3. Distribute a copy of Apache License 2.0 for the BasicSR-derived LayerNorm source.
4. Mark modified/extracted/consolidated files prominently as changed, as required by Apache-2.0 section 4(b).
5. Retain pertinent copyright, patent, trademark, and attribution notices.
6. Do not remove upstream source headers or suggest that UIE3 authored the imported implementation.

The complete composite upstream `LICENSE`, rather than a hand-written summary, should be placed at `UIE3/third_party/nafnet/LICENSE`. Source headers should identify both the upstream origin and any UIE3 import-path/consolidation modifications. This audit is an engineering/license inventory, not legal advice.

A later `UIE3/third_party/nafnet/UPSTREAM.md` must include at least:

- upstream repository URL: `https://github.com/megvii-research/NAFNet`
- pinned upstream commit: `2b4af71ebe098a92a75910c233a3965a3e93ede4`
- acquisition date: 2026-07-18
- exact upstream paths and imported symbol list
- destination file list
- every modification, including symbol extraction, consolidation, formatting, and import rewrites
- removed BasicSR and Local framework dependencies
- a statement of whether computation semantics changed (target: no)
- upstream and vendored license locations
- applicable megvii-model and BasicSR copyrights
- commands/environment and results of official-versus-imported consistency tests

`UPSTREAM.md` must be updated if the pinned commit or imported symbols change. Official copyright notices must never be deleted.

## 13. Planned Equivalence Tests

No test file or executable model was created in Phase A. The following plan is for a later authorized phase.

### Common test setup

1. Instantiate the official ordinary `NAFNet` from the pinned upstream checkout and the imported UIE3 implementation with exactly:
   - `img_channel=3`
   - `width=32`
   - `enc_blk_nums=[2, 2, 2]`
   - `middle_blk_num=4`
   - `dec_blk_nums=[2, 2, 2]`
2. Seed initialization once, take one model’s `state_dict`, and load that same mapping into the other with `strict=True`. Do not compare independently initialized weights.
3. Put both models on the same device with the same dtype and call `eval()` on both.
4. Disable augmentation and stochasticity. For the strict reference test, prefer CPU float32, a fixed PyTorch version, `torch.no_grad()`, a fixed random seed, and deterministic algorithms. Do not use AMP, TF32, `torch.compile`, or different backends in the reference comparison.
5. Run identical random inputs of at least:
   - `[1, 3, 256, 256]`
   - `[2, 3, 128, 192]`
   - `[1, 3, 257, 259]`
6. For every input compare:
   - exact output shape
   - maximum absolute error
   - mean absolute error
   - total and trainable parameter counts
   - ordered `state_dict` keys, tensor shapes, and dtypes
   - strict load result (no missing or unexpected keys)
   - padding multiple and padded shape
   - right/bottom zero-padding behavior and final crop to original `H, W`
7. Acceptance thresholds on the same device, dtype, PyTorch build, and operator path:
   - `max_abs_error <= 1e-6`
   - `mean_abs_error <= 1e-7`

Different devices, mixed precision, TF32, or nondeterministic kernels may introduce different floating-point accumulation. Such a run must be reported separately and cannot silently replace the strict float32 reference. Any wider tolerance must be justified by measured backend/dtype error after the strict implementation-equivalence test passes; errors must not be ignored.

For `[1, 3, 257, 259]`, the research model’s multiple is 8, so the expected internal spatial size is `[264, 264]`, followed by a crop back to `[257, 259]`.

### Planned test cases

- `test_official_and_imported_output_equivalence`: perform the shared-state, three-shape comparison and enforce both error thresholds.
- `test_nafnet_small_output_shape`: assert `[B,3,H,W] -> [B,3,H,W]` for multiple-sized inputs.
- `test_nafnet_small_non_multiple_size`: assert right/bottom zero padding to a multiple of 8 and exact crop for `257 x 259` (and preferably additional small odd sizes).
- `test_nafnet_small_forward_backward`: run a finite scalar loss and backward pass; assert finite input/parameter gradients and, for the official/imported pair, compare gradients under the same state/input where practical. `eval()` does not disable autograd.
- `test_nafnet_small_state_dict_compatibility`: assert identical keys/shapes/counts and bidirectional `strict=True` loading.
- `test_nafnet_small_global_residual_not_duplicated`: set the ending convolution weight and bias to zero, use a nonzero input, and assert output equals the input after crop. An erroneous wrapper-level second residual would instead produce an additional copy and fail. Also compare directly against the official model.

The test harness should load the upstream module in an isolated namespace/path so it cannot accidentally import the UIE3 class under the official name. It must record the upstream SHA, PyTorch version, device, dtype, seed, and deterministic settings with its results.

## 14. Blocking Unknowns

### Minimal-import decision

No blocking unknown remains for the narrowly scoped minimal ordinary-NAFNet import:

- the official core implementation is located;
- the composite license and copyrights are explicit;
- the UIE3 tree has no conflicting structure or same-name implementation;
- the research `NAFNet-small` configuration is explicit and its difference from all inspected official width-32 recipes is resolved by the protocol;
- input/output, padding, crop, and one global residual are unambiguous;
- a strict output/state/gradient consistency test can be implemented later.

### Unknowns that still block later experimental/training work

UIE3 currently has no implementation infrastructure, and the state file already identifies unresolved dataset split, image range, PSNR/SSIM definitions, loss, checkpoint selection/resume, and entry points. The Python/PyTorch packaging and supported runtime are also not declared in UIE3. These facts do not change the five-symbol architecture import decision, but they must be resolved before Phase B integration/testing and certainly before training. This audit does not resolve them by importing BasicSR defaults.

Human acceptance of Phase A remains mandatory. Phase B is not authorized.

## 15. Recommended Phase-2 File Changes

After human acceptance and explicit authorization of the next implementation phase, the smallest recommended change set is:

1. Create `third_party/nafnet/nafnet_arch.py` containing only the five audited symbols, exact forward/backward math, exact public defaults/names, preserved notices, and documented import-only modifications.
2. Copy the complete upstream composite `LICENSE` to `third_party/nafnet/LICENSE`.
3. Create `third_party/nafnet/UPSTREAM.md` with the provenance fields specified in section 12.
4. Create `src/models/backbones/nafnet_small.py` as a thin constructor/wrapper for the declared research configuration. It must return the official class output directly and must not add a second global residual or clamp.
5. Create `tests/test_nafnet_import.py` with the six planned tests in section 13, pinned to the audited upstream SHA.
6. Add only the package initializers/import exposure required by the subsequently approved UIE3 packaging convention.

Do not add `NAFNetLocal`, `Local_Base`, BasicSR training/data/registry code, official task YAMLs, pretrained weights, color/scattering operators, routing models, or training code as part of the minimal import. This list is a recommendation only; no listed file was created in Phase A.

## 16. Final Audit Decision

**READY_FOR_MINIMAL_IMPORT**

The decision means only that the ordinary NAFNet dependency boundary, research configuration, forward semantics, provenance obligations, non-import list, UIE3 destination layout, and future equivalence criteria are sufficiently defined for a later human-authorized minimal import. It does not mean an implementation exists, tests have run, Phase A has been human-accepted, or Phase B is authorized.

The upstream repository remained read-only. Phase A changed only `UIE3/NAFNET_IMPORT_AUDIT.md`; it did not modify the protocol, decisions, state, or any source/configuration file. Work must stop here pending human review.

本阶段只完成了 NAFNet 最小导入审计，尚未复制或实现任何 NAFNet、NAFNet-small、颜色算子、散射算子或训练代码。
