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

---

## Decision D-0011

- Date: 2026-07-18
- Status: Accepted
- Decision: Accept Phase B1b cloud runtime validation.
- Phase B1b status: PASS
- Evidence:
  - 8 tests passed;
  - max absolute error = 0;
  - mean absolute error = 0;
  - official and imported parameter count = 2,846,755;
  - forward, backward, padding, cropping, state_dict compatibility and
    global residual tests passed.
- Authorized next phase:
  Phase B2a — LSUI dataset and split protocol audit.
- Phase B2a scope:
  - inspect the real LSUI directory;
  - verify paired input/GT integrity;
  - determine exact train/validation/test semantics;
  - freeze a deterministic split protocol;
  - do not implement training code.
- Explicitly not authorized:
  - color operator;
  - scattering operator;
  - order model;
  - router;
  - formal training.
- Scientific conclusion:
  NAFNet import equivalence is verified, but no UIE performance conclusion
  has been obtained.
- Human approver: Repository owner

---

## Decision D-0012

- Date: 2026-07-18
- Status: Accepted
- Decision: Execute Phase B2a as a static audit-tool implementation followed
  by cloud dataset execution and deterministic split generation.
- Context:
  - Codex runs on the local MacBook;
  - LSUI19 exists only in the cloud environment;
  - local Codex cannot directly inspect
    `/root/autodl-tmp/pro/publicdata/LSUI19`.
- Dataset snapshot observed:
  - Train/input: 3851
  - Train/GT: 3851
  - Val/input: 428
  - Val/GT: 428
- Intended split semantics:
  - original Train is the development pool;
  - 3466 samples are used for training;
  - 385 samples are used for validation;
  - original Val is held out as the 428-sample final test set.
- Split seed: 3407
- Pair key: exact filename stem
- Images will not be copied or moved.
- Required artifacts:
  - reproducible audit tool;
  - machine-readable audit result;
  - fixed train/validation/test manifests;
  - dataset audit report.
- Scientific semantics changed: No.
- Invalidated experiments: None.
- Human approver: Repository owner

---

## Decision D-0014

- Date: 2026-07-19
- Status: Accepted
- Decision: Adopt a duplicate-aware LSUI split while retaining all original
  paired samples.
- Original dataset:
  - root: /root/autodl-tmp/pro/publicdata/LSUI
  - paired samples: 4279
- Physical split:
  - Train development pool: 3851
  - held-out final test: 428
- Duplicate policy:
  - no original sample is deleted;
  - every sample belonging to a repeated decoded-RGB input group or repeated
    decoded-RGB GT group is assigned to the development pool;
  - all such duplicated samples must remain in the formal training subset;
  - validation is selected only from globally non-duplicated development
    samples.
- Formal experiment split:
  - train: 3466
  - validation: 385
  - test: 428
- Split seed: 3407
- Test-set use:
  - the 428-sample test set must not be used for checkpoint selection,
    hyperparameter selection, router training, or model selection.
- Scientific semantics changed:
  The split protocol is strengthened to prevent duplicate-content leakage.
- Invalidated experiments:
  Any experiment based on the earlier LSUI19 random split.
- Human approver: Repository owner


---

## Decision D-0015

- Date: 2026-07-20
- Status: Accepted
- Decision: Authorize the Phase B2a forced-formal-train manifest.
- Authorized artifact:
  - UIE3/splits/lsui19/forced_formal_train.tsv
- Purpose:
  - record every development-pool sample belonging to a duplicate input
    or duplicate GT content group;
  - verify that all such samples are assigned exclusively to the formal
    training subset;
  - support reproducible leakage checks.
- Scientific semantics changed: No.
- Protocol version change required: No.
- Invalidated experiments: None.
- Current phase remains: B2a.
- Next phase authorized: No.
- Human approver: Repository owner

---

## Decision D-0016

- Date: 2026-07-21
- Status: Accepted
- Decision: Accept Phase B2a duplicate-aware LSUI dataset audit and formal
  split generation.
- Phase B2a status: PASS
- Accepted dataset protocol:
  - physical development pool: 3851 pairs;
  - physical held-out test set: 428 pairs;
  - formal train set: 3466 pairs;
  - formal validation set: 385 pairs;
  - formal test set: 428 pairs;
  - forced formal train samples: 112;
  - total manifest coverage: 4279 unique samples;
  - cross-split decoded-RGB input overlap: 0;
  - cross-split decoded-RGB GT overlap: 0.
- Authorized next phase:
  Phase B2b — static implementation of the NAFNet-small baseline data,
  training, checkpoint and evaluation infrastructure.
- Phase B2b may implement:
  - manifest-based paired image dataset;
  - paired crop and augmentation;
  - NAFNet-small baseline construction;
  - reconstruction loss;
  - RGB PSNR and SSIM;
  - baseline training entry point;
  - baseline evaluation entry point;
  - checkpoint save/load/resume;
  - static and runtime test definitions.
- Phase B2b must not:
  - run formal training;
  - use the final test set for model selection;
  - implement color correction;
  - implement scattering removal;
  - implement order comparison;
  - implement Oracle analysis;
  - implement routing.
- Scientific semantics changed: No.
- Invalidated experiments: None.
- Human approver: Repository owner

---

## Decision D-0017

- Date: 2026-07-21
- Status: Accepted
- Decision: Accept Phase B2b static implementation of the NAFNet-small
  baseline training infrastructure.
- Phase B2b status: IMPLEMENTED_NOT_RUNTIME_VALIDATED
- Authorized next phase:
  Phase B2c — cloud runtime validation and real-data smoke testing.
- Phase B2c scope:
  - run all unit and integration tests in the cloud environment;
  - load real LSUI samples from the formal manifests;
  - execute forward, backward, and optimizer steps;
  - verify AMP behavior on CUDA;
  - run a short 1–4 sample overfit smoke test;
  - verify checkpoint save, load, and resume;
  - generate per-image validation PSNR and SSIM on a temporary validation
    subset;
  - verify that test-split evaluation is rejected.
- Explicitly not authorized:
  - formal baseline training;
  - final test-set evaluation;
  - color correction operator;
  - scattering removal operator;
  - order comparison;
  - Oracle analysis;
  - adaptive routing.
- Scientific semantics changed: No.
- Invalidated experiments: None.
- Human approver: Repository owner


---

## Decision D-0018

- Date: 2026-07-21
- Status: Accepted
- Decision: Remediate the baseline CLI manifest guard discovered during
  Phase B2c runtime validation.
- Observed failure:
  - Phase B2c requires a 1–4 sample real-data smoke test;
  - the baseline CLI currently requires the exact canonical manifest path
    `splits/lsui19/train.tsv`;
  - therefore a temporary smoke-test manifest cannot be supplied.
- Remediation:
  - retain canonical-manifest enforcement for normal training;
  - add an explicit smoke-test mode;
  - allow temporary train and validation manifest overrides only when
    smoke-test mode is explicitly enabled;
  - require a finite positive `--max-steps` in smoke-test mode;
  - continue to reject test-split evaluation;
  - log that the run is not a formal experiment.
- Formal manifests must not be modified.
- Formal experiment semantics changed: No.
- Invalidated experiments: None.
- Current runtime validation remains incomplete.
- Human approver: Repository owner

---

## Decision D-0019

- Date: 2026-07-22
- Status: Accepted
- Decision: Resume Phase B2c for report-only closure after completing the
  smoke-manifest guard remediation and cloud runtime validation.
- Current phase:
  Phase B2c — baseline runtime and real-data smoke validation.
- Current remaining work:
  - generate reports/PHASE_B2C_REPORT.md from committed runtime evidence;
  - perform human acceptance.
- Writable scope:
  - UIE3/reports/PHASE_B2C_REPORT.md only.
- Runtime experiments must not be rerun in this report-only closure step.
- Source code, tests, configs, manifests and runtime evidence are read-only.
- Next phase authorized: No.
- Scientific semantics changed: No.
- Invalidated experiments: None.
- Human approver: Repository owner

---

## Decision D-0020

- Date: 2026-07-22
- Status: Accepted
- Decision: Accept Phase B2c cloud runtime and real-data smoke validation.
- Phase B2c status: PASS
- Accepted evidence:
  - pytest: 30 passed, 0 failed, exit code 0;
  - 161 warnings were recorded and did not cause test failure;
  - real LSUI CUDA smoke training completed;
  - 40 continuous optimizer steps were completed;
  - checkpoint resume advanced global_step from 20 to 40;
  - model and optimizer states were restored;
  - validation exported four per-image RGB PSNR and SSIM records;
  - all validation metrics were finite;
  - final test evaluation was rejected with exit code 2;
  - no forbidden test CSV was generated.
- Authorized next phase:
  Phase B3a — formal NAFNet-small baseline configuration freeze and resource
  validation.
- Phase B3a scope:
  - verify the candidate formal configuration on the complete formal train
    and validation manifests;
  - measure CUDA memory, training throughput, validation runtime, checkpoint
    size and estimated total training cost;
  - freeze the formal baseline configuration and checkpoint-selection rule.
- Phase B3a must not:
  - launch full 200-epoch training;
  - evaluate the final test set;
  - implement color correction;
  - implement scattering removal;
  - implement operator ordering;
  - implement Oracle analysis or routing.
- Scientific semantics changed: No.
- Invalidated experiments: None.
- Human approver: Repository owner

---

## Decision D-0021

- Date: 2026-07-22
- Status: Accepted
- Decision: Resolve the Phase B3a configuration revision by freezing the
  official formal-training hardware and training-budget policy.
- Phase B3a runtime-chain result:
  - the formal train manifest loaded successfully;
  - 100 CUDA optimizer steps completed;
  - batch_size=4 and patch_size=256 completed without CUDA OOM;
  - checkpoint saving completed;
  - the complete 385-sample validation path completed;
  - all per-image RGB PSNR and SSIM values were finite;
  - the final test split was not evaluated.
- Cause of the previous NEEDS_CONFIGURATION_REVISION status:
  - batch_size=4 used approximately 17148 MiB on an RTX 3090;
  - this exceeds the previously assumed RTX 3060 12GB portability target;
  - no runtime or scientific failure was observed on the actual cloud
    execution platform.
- Hardware policy:
  - the official formal experiment platform is an NVIDIA RTX 3090 24GB or
    an equivalent GPU with sufficient memory;
  - the local RTX 3060 12GB is a development and lightweight smoke-test
    platform;
  - fitting the complete formal training configuration into 12GB is not a
    formal acceptance requirement.
- Frozen resource configuration:
  - patch_size: 256;
  - batch_size: 4;
  - num_workers: 4;
  - AMP: true.
- Frozen optimization protocol:
  - optimizer: AdamW;
  - learning rate: 2.0e-4;
  - weight decay: 0;
  - maximum training length: 200 epochs;
  - checkpoint selection: highest formal-validation RGB PSNR;
  - the final test split must not participate in checkpoint selection.
- The 200-epoch value is a fixed maximum training budget and is not claimed
  to be performance-optimal.
- Formal seeds remain:
  - 3407;
  - 1234;
  - 2027.
- Estimated compute:
  - approximately 20.14 hours per seed on the measured RTX 3090 setup;
  - approximately 60.43 hours for three seeds, excluding scheduling and
    additional operational overhead.
- Current phase:
  Phase B3a-r1 — report-only hardware and training-budget freeze.
- Next phase authorized: No.
- Scientific semantics changed: No.
- Invalidated experiments: None.
- Human approver: Repository owner

---

## Decision D-0022

- Date: 2026-07-22
- Status: Accepted
- Decision: Accept Phase B3a-r1 formal hardware and training-budget freeze.
- Phase B3a-r1 status: PASS
- Frozen formal-training platform:
  - NVIDIA RTX 3090 24GB or an equivalent GPU with sufficient memory.
- Frozen model and optimization protocol:
  - NAFNet-small;
  - patch_size: 256;
  - batch_size: 4;
  - num_workers: 4;
  - AMP: true;
  - optimizer: AdamW;
  - learning rate: 2.0e-4;
  - weight decay: 0;
  - maximum training length: 200 epochs;
  - checkpoint selection: highest formal-validation RGB PSNR.
- Frozen formal seed set:
  - 3407;
  - 1234;
  - 2027.
- Operational checkpoint policy:
  - best.pt and last.pt must be retained;
  - periodic epoch checkpoints may be retained every 10 epochs;
  - periodic checkpoint retention does not alter optimization or model
    selection semantics.
- Authorized next phase:
  Phase B3b-1 — formal NAFNet-small baseline training for seed 3407 only.
- Phase B3b-1 scope:
  - train seed 3407 for the frozen maximum budget;
  - use only the formal train manifest for optimization;
  - use only the formal validation manifest for checkpoint selection;
  - export per-image validation metrics for the selected best checkpoint;
  - preserve complete configuration, logs and checkpoint provenance.
- Explicitly not authorized:
  - seed 1234 training;
  - seed 2027 training;
  - final test evaluation;
  - color correction operator;
  - scattering removal operator;
  - operator-order comparison;
  - Oracle analysis;
  - adaptive routing.
- Scientific semantics changed: No.
- Invalidated experiments: None.
- Human approver: Repository owner


---

## Decision D-0023

- Date: 2026-07-22
- Status: Accepted
- Decision: Remediate AMP overflow handling discovered during formal
  seed-3407 baseline training.
- Observed failure:
  - seed 3407 completed four validation epochs;
  - the last completed global_step was 3468;
  - training then stopped because an unscaled parameter gradient was
    non-finite;
  - train loss and validation metrics were finite before the failure.
- Root cause:
  - the trainer calls a fatal finite-gradient assertion after
    GradScaler.unscale_ and before GradScaler.step;
  - this prevents GradScaler from skipping an overflowed optimizer update
    and reducing its scale as intended.
- Required remediation:
  - AMP overflow must cause the affected optimizer update to be skipped;
  - AMP scale must be updated normally;
  - skipped optimizer updates must not increment global_step;
  - non-AMP training must retain strict non-finite-gradient failure;
  - finite-gradient clipping behavior must remain unchanged;
  - overflow events must be observable in returned training-step metadata
    and logs.
- Frozen scientific configuration remains unchanged:
  - model architecture;
  - batch_size;
  - patch_size;
  - optimizer;
  - learning rate;
  - loss;
  - formal manifests;
  - seed;
  - maximum epochs.
- Existing seed-3407 checkpoints and failure evidence must be preserved.
- Formal seed-3407 training may resume only after remediation tests and a
  cloud runtime validation pass.
- Next phase authorized: No.
- Scientific semantics changed: No.
- Invalidated completed experiments: None.
- Human approver: Repository owner

---

## Decision D-0024

- Date: 2026-07-22
- Status: Accepted
- Decision: Accept the static implementation of the AMP overflow handling
  remediation and authorize cloud runtime validation.
- Static remediation status:
  IMPLEMENTED_NOT_CLOUD_RUNTIME_VALIDATED
- Required runtime validation:
  - run the complete pytest suite in the cloud PyTorch environment;
  - run an explicit CUDA AMP overflow regression;
  - verify that an overflowed optimizer update is skipped;
  - verify that global_step is not incremented for a skipped update;
  - verify that AMP scale decreases after overflow;
  - verify that a later finite update succeeds;
  - verify checkpoint compatibility using the interrupted seed-3407
    checkpoint in a temporary resume probe;
  - do not modify the existing interrupted formal run.
- Existing interrupted run:
  - remains preserved as failure evidence;
  - is not a completed formal baseline experiment;
  - must not be resumed as the final formal run.
- Formal configuration remains unchanged.
- Formal seed-3407 training is not yet authorized.
- Next phase authorized: No.
- Scientific semantics changed: No.
- Invalidated completed experiments: None.
- Human approver: Repository owner

---

## Decision D-0025

- Date: 2026-07-23
- Status: Accepted
- Decision: Accept Phase B3b-1r1c cloud runtime validation of the AMP
  overflow remediation.
- Phase B3b-1r1c status: PASS
- Accepted evidence:
  - the complete cloud pytest suite passed;
  - a real CUDA AMP overflowed optimizer update was skipped;
  - model parameters did not change during the skipped update;
  - global_step did not increase during the skipped update;
  - AMP scale decreased after the overflow;
  - a subsequent finite optimizer update succeeded;
  - the interrupted seed-3407 checkpoint was strictly restored in a
    temporary resume probe;
  - model, optimizer and GradScaler states were present;
  - final test was not evaluated.
- Scientific configuration remains frozen:
  - NAFNet-small;
  - seed: 3407;
  - patch_size: 256;
  - batch_size: 4;
  - num_workers: 4;
  - AMP: true;
  - AdamW;
  - learning rate: 2.0e-4;
  - weight decay: 0;
  - maximum training length: 200 epochs;
  - checkpoint selection: highest formal-validation RGB PSNR.
- Authorized next phase:
  Phase B3b-1r2 — restart formal seed-3407 training from epoch 0 using the
  remediated trainer.
- Formal-run policy:
  - use the single remediated Git commit for the complete run;
  - do not resume the interrupted pre-remediation run;
  - use a new output directory;
  - preserve the old interrupted run as diagnostic evidence;
  - use only formal train and validation manifests;
  - final test must not be evaluated.
- New formal output directory:
  /root/autodl-tmp/pro/UIE3_runs/b3b/seed_3407_r1
- Explicitly not authorized:
  - seed 1234 training;
  - seed 2027 training;
  - final test evaluation;
  - source-code changes;
  - formal-config changes;
  - color or scattering operators;
  - operator-order experiments;
  - Oracle analysis;
  - adaptive routing.
- Scientific semantics changed: No.
- Invalidated completed experiments: None.
- Human approver: Repository owner

---

## Decision D-0026

- Date: 2026-07-23
- Status: Accepted
- Decision: Accept the completed formal NAFNet-small baseline run for seed
  3407.
- Phase B3b-1r2 status: PASS
- Accepted seed-3407 evidence:
  - the run started from epoch 0 using the remediated trainer;
  - 200 epochs, indexed 0 through 199, completed;
  - train exit code was 0;
  - the interrupted pre-remediation run was not resumed;
  - best.pt was selected using the highest formal-validation RGB PSNR;
  - best checkpoint: epoch 192, global_step 167267;
  - last checkpoint: epoch 199, global_step 173334;
  - the complete 385-sample formal validation set was evaluated;
  - mean RGB PSNR: 27.319515354602366;
  - mean RGB SSIM: 0.8942945567044345;
  - all per-image metrics were finite;
  - final test was not evaluated.
- Interpretation:
  - this is the completed formal validation result for seed 3407 only;
  - it is not yet the three-seed baseline result.
- Authorized next phase:
  Phase B3b-2 — formal NAFNet-small baseline training for seed 1234.
- Frozen protocol:
  - model, loss, optimizer, learning rate, data manifests, augmentation,
    batch size, patch size, AMP and maximum epoch budget remain unchanged;
  - checkpoint selection remains highest formal-validation RGB PSNR;
  - final test must not participate in training or checkpoint selection.
- Explicitly not authorized:
  - seed 2027 training;
  - final test evaluation;
  - source-code changes;
  - formal-protocol changes;
  - color or scattering operators;
  - operator-order experiments;
  - Oracle analysis;
  - adaptive routing.
- Scientific semantics changed: No.
- Invalidated experiments: None.
- Human approver: Repository owner