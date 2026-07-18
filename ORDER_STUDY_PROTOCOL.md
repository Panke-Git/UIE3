# ORDER_STUDY_PROTOCOL

## 1. Document status

- Protocol name: Adaptive Non-Commutative Operator Ordering for Underwater Image Enhancement
- Protocol version: 1.0.1
- Repository: `UIE3_workspace/UIE3`
- Read-only upstream repository: `UIE3_workspace/NAFNet`
- Current authorized phase: Defined exclusively by `ORDER_STUDY_STATE.yaml`
- Primary hardware target: NVIDIA RTX 3060 12 GB
- Intended venue: ICASSP 2027
- Status: Active research protocol

This document is the canonical scientific and engineering specification for the project. Codex may implement only the phase explicitly authorized in `ORDER_STUDY_STATE.yaml`. Any change that affects the scientific meaning of an experiment must be recorded in `ORDER_STUDY_DECISIONS.md` and must increment `protocol_version`.

`ORDER_STUDY_PROTOCOL.md` is phase-independent. It defines stable scientific
and engineering invariants only.

The current phase, writable paths, required outputs, and advancement
authorization are defined exclusively by `ORDER_STUDY_STATE.yaml`.

If this protocol contains a historical phase example that conflicts with
`ORDER_STUDY_STATE.yaml`, the historical example must be corrected before
execution. Codex must not resolve such a conflict autonomously.

---

## 2. Research objective

The project studies whether two constrained underwater restoration operators are order-sensitive and whether the preferred order is sample-dependent.

Definitions:

- Degraded underwater image:
  \[
  x \in \mathbb{R}^{B\times3\times H\times W}
  \]

- Paired target:
  \[
  y \in \mathbb{R}^{B\times3\times H\times W}
  \]

- Color correction operator:
  \[
  C_{\theta_C}
  \]

- Scattering removal operator:
  \[
  S_{\theta_S}
  \]

- NAFNet-small restoration backbone:
  \[
  B_{\theta_B}
  \]

The two explicit orders are:

### color_then_scatter

\[
F_{C\rightarrow S}(x)
=
B_{\theta_B}
\left(
S_{\theta_S}
\left(
C_{\theta_C}(x)
\right)
\right)
\]

Execution:

```text
x
→ ColorCorrectionOperator
→ ScatteringRemovalOperator
→ NAFNet-small
→ output
```

### scatter_then_color

\[
F_{S\rightarrow C}(x)
=
B_{\theta_B}
\left(
C_{\theta_C}
\left(
S_{\theta_S}(x)
\right)
\right)
\]

Execution:

```text
x
→ ScatteringRemovalOperator
→ ColorCorrectionOperator
→ NAFNet-small
→ output
```

Code, configs, logs, CSV files, and reports must use the full names `color_then_scatter` and `scatter_then_color`. Bare abbreviations such as `CS` and `SC` are prohibited unless their meanings are defined immediately beside them.

---

## 3. Scientific hypotheses

### H1: Empirical non-commutativity

For the constrained operator families implemented in this project, the empirical commutator is generally nonzero:

\[
K(x)
=
S_{\theta_S}(C_{\theta_C}(x))
-
C_{\theta_C}(S_{\theta_S}(x)).
\]

A nonzero commutator establishes only that the two orders produce different intermediate outputs. It does not establish that adaptive ordering improves restoration.

### H2: Sample-dependent preferred order

For paired data, different images may prefer different orders under a predeclared per-image RGB PSNR criterion.

### H3: Deployable order prediction

A lightweight selector that sees only the degraded input may predict the preferred order and improve over the best fixed order without using GT, test metrics, Oracle labels, or duplicated backbones at inference.

---

## 4. Required research sequence

The project must follow this sequence:

```text
Phase A: repository and NAFNet import audit
→ Phase B: NAFNet-small baseline and operator smoke tests
→ Phase C: shared-weight order hypothesis validation
→ Phase D: OOF route labels and route-predictability gate
→ Phase E: adaptive hard-order model and ablations
→ Phase F: full experiments and paper-ready reports
```

No phase may begin automatically. Human acceptance of the previous phase is required.

---

## 5. Global hard invariants

The following rules apply to every phase:

1. `UIE3_workspace/NAFNet` is read-only.
2. Only `UIE3_workspace/UIE3` may be modified.
3. Do not overwrite or destructively modify existing models.
4. Reuse the existing UIE3 training, testing, metric, checkpoint, logging, and configuration systems whenever possible.
5. Do not create a second incompatible training framework.
6. Do not silently guess facts that affect:
   - data split;
   - input range;
   - normalization;
   - metric definition;
   - checkpoint selection;
   - model semantics;
   - experiment fairness.
7. A blocking unknown must stop the current phase.
8. Do not use test GT, test PSNR, test SSIM, or test Oracle labels for training, hyperparameter selection, checkpoint selection, or routing.
9. Do not change the dataset split, main reconstruction loss, data augmentation, metric implementation, checkpoint rule, or NAFNet-small structure while comparing operator order.
10. Do not duplicate NAFNet in the main adaptive model.
11. Soft output fusion is an ablation only and must not be described as hard execution-order selection.
12. Learned transmission, background light, and color matrices are constrained latent variables, not true physical parameters.
13. Do not add GANs, diffusion models, Transformers, external depth models, or large pretrained encoders in the first version.
14. Do not launch multi-hour training unless the current phase explicitly authorizes it.
15. Do not fabricate results.
16. On NaN, Inf, shape mismatch, data leakage, or protocol violation, stop and report the failure.
17. Every experiment must record:
    - git commit;
    - full config;
    - command;
    - random seed;
    - environment;
    - GPU;
    - start/end time;
    - parameter count;
    - output directory.
18. Every accepted phase must have a report and a Git commit/tag.

---

## 6. Workspace contract

Expected workspace:

```text
UIE3_workspace/
├── NAFNet/   # official upstream, read-only
└── UIE3/     # user repository, writable
```

Codex must validate the paths before acting.

Writable paths are phase-specific and are defined exclusively in
`ORDER_STUDY_STATE.yaml` under:

- `phase_permissions.writable_paths`

Required outputs are phase-specific and are defined exclusively in
`ORDER_STUDY_STATE.yaml` under:

- `phase_outputs.required`

The current Codex prompt may narrow these permissions but may not expand
them.

The following governance files may be modified only by explicit human
authorization:

- `ORDER_STUDY_PROTOCOL.md`
- `ORDER_STUDY_DECISIONS.md`
- `ORDER_STUDY_STATE.yaml`

Codex must not update the current phase, authorize the next phase, or alter
the research protocol automatically.

---

## 7. NAFNet import policy

The official NAFNet repository is an upstream reference, not the main project.

The preferred integration policy is:

1. Audit the upstream dependency graph.
2. Import only the minimal code required for ordinary NAFNet.
3. Do not import:
   - NAFNetLocal;
   - Local_Base;
   - complete BasicSR training framework;
   - GoPro/SIDD datasets;
   - official experiment scripts;
   - pretrained weights;
   - unrelated demos.
4. Preserve upstream license and attribution.
5. Record upstream commit and modifications in `third_party/nafnet/UPSTREAM.md`.
6. Verify numerical equivalence between the official and imported implementations.

Suggested layout, subject to the existing UIE3 structure:

```text
UIE3/
├── third_party/
│   └── nafnet/
│       ├── nafnet_arch.py
│       ├── LICENSE
│       └── UPSTREAM.md
├── src/
│   └── models/
│       └── backbones/
│           └── nafnet_small.py
└── tests/
    └── test_nafnet_import.py
```

---

## 8. NAFNet-small definition

`NAFNet-small` is a research configuration of the official NAFNet class. It is not assumed to be an official pretrained model name.

Fallback configuration:

```yaml
img_channel: 3
width: 32
enc_blk_nums: [2, 2, 2]
middle_blk_num: 4
dec_blk_nums: [2, 2, 2]
```

Rules:

1. If UIE3 already contains an explicit NAFNet-small configuration, Codex must report the exact difference before choosing either version.
2. Do not silently replace an existing configuration.
3. Preserve the official NAFNet global residual behavior.
4. Do not add another whole-network residual connection.
5. Support arbitrary image sizes through internal padding and exact output cropping.
6. Input and output must both be `[B,3,H,W]`.

---

## 9. ColorCorrectionOperator specification

The first implementation must be lightweight and globally constrained.

\[
C_{\theta_C}(x)=M(x)x+b(x)
\]

with:

\[
M\in\mathbb{R}^{B\times3\times3},
\qquad
b\in\mathbb{R}^{B\times3\times1\times1}.
\]

Near-identity parameterization:

\[
M=I+\alpha\tanh(\Delta M),
\qquad
b=\beta\tanh(\Delta b).
\]

Conservative defaults:

```yaml
color_hidden: 16
color_matrix_scale: 0.10
color_bias_scale: 0.05
```

Requirements:

1. Final prediction layer zero-initialized.
2. Initial output close to input.
3. One global color matrix per image.
4. No per-pixel 3×3 matrices.
5. No deep U-Net, attention block, or large spatial branch.
6. No hard clamp in the training forward pass.
7. Optional auxiliary outputs:
   - corrected;
   - color_matrix;
   - color_bias.
8. Check finite values, shapes, and configured bounds.

---

## 10. ScatteringRemovalOperator specification

Use a stable physical-inspired inverse:

\[
J_{\mathrm{raw}}
=
\frac{I-A\odot(1-t)}{t+\varepsilon}.
\]

Shapes:

\[
t\in\mathbb{R}^{B\times1\times H\times W},
\qquad
A\in\mathbb{R}^{B\times3\times1\times1}.
\]

Stable residual output:

\[
\Delta=J_{\mathrm{raw}}-I,
\]

\[
J
=
I+r_{\max}\tanh\left(\frac{\Delta}{r_{\max}}\right).
\]

Conservative defaults:

```yaml
scattering_hidden: 16
transmission_min: 0.20
initial_transmission: 0.95
initial_background_light: 0.50
scattering_eps: 1.0e-6
scattering_max_residual: 0.10
```

Requirements:

1. Single-channel transmission.
2. Global RGB background light.
3. Near-identity initialization.
4. No depth input or external network.
5. No claim that learned variables are true physical values.
6. Optional auxiliary outputs:
   - restored;
   - transmission;
   - background_light;
   - raw_inverse.
7. Check denominator, finite values, bounds, and shapes.

---

## 11. Phase B minimum models

Phase B may implement only:

```text
baseline
color_only
scatter_only
```

Phase B must not implement dual-order training, Oracle analysis, OrderSelector, or adaptive routing.

Required engineering checks:

- forward/backward;
- arbitrary H/W;
- NAFNet padding/cropping;
- identity initialization;
- transmission and background-light bounds;
- AMP;
- checkpoint save/load/resume;
- one-batch real-data test;
- 20–100 iteration overfit test on 1–4 samples;
- no long training.

---

## 12. Shared-order diagnostic model

Phase C must use exactly one stored instance of:

- ColorCorrectionOperator;
- ScatteringRemovalOperator;
- NAFNet-small.

Both orders share all parameters.

Joint diagnostic loss:

\[
L_{\mathrm{diag}}
=
\frac12 L(F_{C\rightarrow S}(x),y)
+
\frac12 L(F_{S\rightarrow C}(x),y).
\]

The shared diagnostic model is the primary instrument for H1 and H2.

Separately trained fixed-order models may be used as performance baselines, but they cannot be the sole evidence for order causality because their optimized parameters differ.

---

## 13. Oracle protocol

The unique Oracle path-selection criterion is per-image RGB PSNR.

For each image:

\[
r_i^\*
=
\arg\max_{r\in\{C\rightarrow S,S\rightarrow C\}}
\operatorname{PSNR}(F_r(x_i),y_i).
\]

Formal analysis must first average each path's per-image metric across seeds.

Ambiguous samples are samples satisfying either condition:

1. route sign is inconsistent across seeds;
2. absolute seed-averaged PSNR difference is below `route_margin_db`.

Default:

```yaml
route_margin_db: 0.05
```

Ambiguous samples must not be forced into hard router labels by default.

Required reports:

- fixed-best PSNR;
- Oracle PSNR;
- Oracle gap;
- preference ratios;
- ambiguous ratio;
- cross-seed route agreement;
- paired bootstrap 95% confidence interval;
- Wilcoxon signed-rank test;
- top-5% contribution concentration.

---

## 14. Order hypothesis gate

Default thresholds:

```yaml
minimum_oracle_gap_db: 0.10
minimum_minority_route_ratio: 0.15
minimum_cross_seed_agreement: 0.67
route_margin_db: 0.05
require_bootstrap_ci_lower_gt_zero: true
maximum_top5_contribution_ratio: 0.50
```

Gate states:

- `PASS`: adaptive routing may proceed.
- `WEAK`: exploration only; no main claim.
- `FAIL`: adaptive routing blocked.

No OrderSelector may be implemented before the shared-order diagnostic code and gate are complete.

---

## 15. OOF route-label protocol

Router training labels must be out-of-fold.

Development setting:

```yaml
oof_folds: 3
```

Recommended final setting:

```yaml
oof_folds: 5
```

Every router-training sample must receive its label from a restoration model that did not train on that sample.

Test labels are analysis-only and must never be loaded by router training code.

---

## 16. Route-predictability gate

The selector sees only the degraded image.

Forbidden selector inputs:

- GT;
- PSNR;
- SSIM;
- reconstruction errors;
- branch outputs;
- Oracle metrics;
- test-only information.

Default gate thresholds:

```yaml
minimum_balanced_accuracy_gain_over_majority: 0.05
minimum_minority_recall: 0.40
minimum_restoration_gain_db: 0.05
```

The route-predictability gate must evaluate:

- majority baseline;
- explicit-statistics logistic regression;
- lightweight CNN selector;
- frozen shared restoration model under predicted hard routes.

---

## 17. Final adaptive model

The primary adaptive model contains exactly:

- one ColorCorrectionOperator;
- one ScatteringRemovalOperator;
- one NAFNet-small;
- one lightweight OrderSelector.

Inference:

\[
r=\arg\max R(x).
\]

Only the selected order is executed.

Soft output mixture:

\[
p_0y_{C\rightarrow S}+p_1y_{S\rightarrow C}
\]

is an ablation, not the main method.

Optional ST-Gumbel training must use one-hot forward routing and argmax inference.

---

## 18. Fixed experimental controls

After Phase A confirms the existing UIE3 implementation, the following must be frozen before order comparison:

- train/validation/test split;
- input range;
- normalization;
- crop and augmentation;
- reconstruction loss;
- optimizer;
- scheduler;
- training epochs;
- batch size/effective batch size;
- checkpoint rule;
- PSNR implementation;
- SSIM implementation;
- clamp and crop-border behavior;
- NAFNet-small structure;
- operator definitions and defaults.

Any later change invalidating previous experiments must be recorded in `ORDER_STUDY_DECISIONS.md`, and affected experiments must be rerun.

---

## 19. Reproducibility

Formal fixed-order and diagnostic experiments must support at least three seeds.

Initial planned seeds:

```yaml
seeds: [3407, 1234, 2027]
```

Every formal result must report mean ± standard deviation.

All per-image results must be saved. Reporting only the best seed is prohibited.

---

## 20. Required phase reports

Each phase must produce the report file declared for that phase in
`ORDER_STUDY_STATE.yaml` under `phase_outputs.required`.

The exact report filename is phase-specific. Examples may include
`PHASE_A_REPORT.md`, `PHASE_B1A_REPORT.md`, or `PHASE_C_REPORT.md`, but the
state file is the sole source of truth.

Each report must include:

1. phase objective;
2. starting commit;
3. ending commit;
4. changed files;
5. assumptions;
6. commands;
7. actual test outputs;
8. acceptance criteria;
9. unresolved risks;
10. supported conclusions;
11. unsupported conclusions;
12. whether the next phase is authorized.

---

## 21. Protocol-change rule

Any change to this protocol must:

1. be recorded in `ORDER_STUDY_DECISIONS.md`;
2. increment `protocol_version`;
3. identify invalidated experiments;
4. specify required reruns;
5. be human-approved before Codex continues.

Codex must not modify the research protocol to make an experiment pass.
