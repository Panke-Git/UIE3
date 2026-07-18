# ORDER_STUDY_DECISIONS

This file records human-approved decisions that affect the implementation or scientific interpretation of the order study.

Do not delete earlier entries. Append new entries chronologically.

---

## Decision D-0001

- Date: 2026-07-18
- Status: Accepted
- Decision: Use `UIE3_workspace/UIE3` as the sole writable research repository.
- Read-only upstream: `UIE3_workspace/NAFNet`
- Reason:
  - keep official upstream code separate;
  - avoid nested Git repositories;
  - preserve provenance;
  - allow a minimal, auditable import.
- Alternatives considered:
  - copy the complete NAFNet repository into UIE3;
  - fork NAFNet as the primary project.
- Rejected because:
  - full-copy imports unrelated BasicSR infrastructure;
  - it obscures which code is used;
  - it increases the risk of accidental official-code modification.
- Invalidated experiments: None.
- Required reruns: None.

---

## Decision D-0002

- Date: 2026-07-18
- Status: Accepted
- Decision: The first Codex phase is an audit only.
- Authorized output:
  - `UIE3/NAFNET_IMPORT_AUDIT.md`
- Forbidden during Phase A:
  - source-code copying;
  - model implementation;
  - package installation;
  - training;
  - operator implementation.
- Reason:
  - determine the minimal NAFNet dependency set;
  - confirm license obligations;
  - inspect UIE3 integration points before code changes.
- Invalidated experiments: None.
- Required reruns: None.

---

## Decision D-0003

- Date: 2026-07-18
- Status: Accepted
- Decision: `NAFNet-small` is a research configuration of the official NAFNet class, not an official pretrained model name.
- Fallback configuration:
  - img_channel: 3
  - width: 32
  - enc_blk_nums: [2, 2, 2]
  - middle_blk_num: 4
  - dec_blk_nums: [2, 2, 2]
- Constraint:
  - if UIE3 already has a conflicting `NAFNet-small`, Phase A must report it and request a human decision.
- Invalidated experiments: None.
- Required reruns: None.

---

## Decision D-0004

- Date: 2026-07-18
- Status: Accepted
- Decision: The order hypothesis must be tested with a shared-weight diagnostic model.
- Shared parameters:
  - one ColorCorrectionOperator;
  - one ScatteringRemovalOperator;
  - one NAFNet-small.
- Reason:
  - separately trained fixed-order models differ in optimized parameters;
  - shared parameters better isolate execution order.
- Separately trained fixed-order models:
  - allowed as performance baselines;
  - not sufficient as sole causal evidence.
- Invalidated experiments: Any previous conclusion based only on independently trained fixed-order models.
- Required reruns: Shared-order diagnostic experiment.

---

## Decision D-0005

- Date: 2026-07-18
- Status: Accepted
- Decision: Router labels must be out-of-fold and ambiguous samples must not be forced into hard labels by default.
- Development folds: 3
- Recommended final folds: 5
- Initial route margin: 0.05 dB
- Reason:
  - avoid training-sample memorization;
  - avoid treating negligible PSNR differences as meaningful class labels.
- Invalidated experiments: Any router trained on in-sample teacher labels without explicit ablation status.
- Required reruns: OOF label generation and router retraining.

---

## Decision D-0006

- Date: 2026-07-18
- Status: Accepted
- Decision: Adaptive routing is blocked unless both gates pass.
- Gate 1:
  - order hypothesis gate.
- Gate 2:
  - route-predictability gate.
- Main adaptive model:
  - one C;
  - one S;
  - one NAFNet-small;
  - one lightweight selector;
  - hard route at inference.
- Soft mixture:
  - ablation only.
- Invalidated experiments: None.
- Required reruns: None.

---

# New decision template

## Decision D-XXXX

- Date:
- Status: Proposed / Accepted / Rejected / Superseded
- Decision:
- Context:
- Alternatives considered:
- Reason:
- Files/configs affected:
- Scientific assumptions affected:
- Invalidated experiments:
- Required reruns:
- Human approver:
---

## Decision D-0007

- Date: 2026-07-18
- Status: Accepted
- Decision: Accept the Phase A NAFNet minimal import audit.
- Audit result: READY_FOR_MINIMAL_IMPORT
- Accepted upstream commit:
  `2b4af71ebe098a92a75910c233a3965a3e93ede4`
- Authorized next phase:
  Phase B1 — minimal ordinary NAFNet import and numerical equivalence tests.
- Phase B1 scope:
  - import only the five audited architecture symbols;
  - preserve licenses and provenance;
  - create the project-specific NAFNet-small wrapper;
  - run official-versus-imported equivalence tests.
- Explicitly not authorized:
  - dataset implementation;
  - training framework;
  - color operator;
  - scattering operator;
  - order model;
  - routing;
  - long training.
- Scientific conclusion:
  No experimental or restoration conclusion has been obtained.
- Invalidated experiments: None.
- Required reruns: None.
- Human approver: Repository owner

---

## Decision D-0008

- Date: 2026-07-18
- Status: Accepted
- Decision: Split Phase B1 into a local static implementation phase and a cloud runtime validation phase.
- Reason:
  - Codex runs on the local MacBook environment;
  - the local Python 3.9.6 environment does not contain PyTorch;
  - formal code execution and later experiments will run in the cloud environment;
  - installing or changing dependencies was not authorized in the original Phase B1 prompt.
- Phase B1a:
  - implement the minimal NAFNet import;
  - create the NAFNet-small wrapper;
  - create equivalence tests;
  - perform syntax and static checks only;
  - do not claim runtime equivalence.
- Phase B1b:
  - run the equivalence, forward, backward, padding and residual tests in a cloud environment with PyTorch;
  - record actual runtime results.
- Scientific semantics changed: No.
- Invalidated experiments: None.
- Required reruns: Phase B1 runtime validation must be completed in the cloud.
- Human approver: Repository owner

---

## Decision D-0009

- Date: 2026-07-18
- Status: Accepted
- Decision: Make `ORDER_STUDY_PROTOCOL.md` phase-independent and designate
  `ORDER_STUDY_STATE.yaml` as the sole source of truth for the current phase,
  writable paths, and required outputs.
- Context:
  - Phase B1a was blocked because the protocol still contained Phase A-specific
    authorization language.
  - `phase_permissions.writable_paths` required
    `reports/PHASE_B1A_REPORT.md`, while `phase_outputs.required` still required
    `reports/PHASE_B1_REPORT.md`.
- Resolution:
  - protocol version increased from 1.0.0 to 1.0.1;
  - current phase is no longer hard-coded in the protocol;
  - Phase B1a report path is unified as
    `UIE3/reports/PHASE_B1A_REPORT.md`;
  - `ORDER_STUDY_STATE.yaml` is the sole phase-control source.
- Scientific semantics changed: No.
- Model semantics changed: No.
- Invalidated experiments: None.
- Required reruns: None.
- Authorized phase after this decision:
  Phase B1a — NAFNet minimal import static implementation.
- Human approver: Repository owner

---

## Decision D-0010

- Date: 2026-07-18
- Status: Accepted
- Decision: Accept Phase B1a static NAFNet import implementation.
- Phase B1a status: IMPLEMENTED_NOT_RUNTIME_VALIDATED
- Authorized next phase:
  Phase B1b — cloud runtime equivalence validation.
- Phase B1b scope:
  - run official-versus-imported numerical equivalence tests;
  - run forward and backward tests;
  - validate padding, cropping, state_dict compatibility and global residual;
  - record the actual cloud environment and raw test output.
- Explicitly not authorized:
  - dataset implementation;
  - training framework;
  - UIE training;
  - color operator;
  - scattering operator;
  - order model;
  - routing.
- Scientific conclusion:
  No UIE or order-study conclusion has been obtained.
- Human approver: Repository owner