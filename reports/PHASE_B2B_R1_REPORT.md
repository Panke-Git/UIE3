# Phase B2b-r1 Report — Baseline Smoke Manifest Override Remediation

## Phase Objective

修复 baseline 训练与 validation 评价 CLI 的 manifest 保护规则：正式模式继续强制使用 canonical train/validation manifests；只有显式、有限步数、标记为非正式实验的 smoke 模式可以使用受限 override；最终 test manifest 和 test split 始终禁止。

## Governance Confirmation

- `current_phase.id`: `B2b-r1`
- `current_phase.name`: `baseline_smoke_manifest_override_remediation`
- `current_phase.authorized`: `true`
- `current_phase.next_phase_authorized`: `false`
- 本阶段四个写入路径与 `phase_permissions.writable_paths`、`phase_outputs.required` 一致。
- 未修改治理文件、配置、正式 split manifests、模型、数据模块或数据集图片。

## Starting and Ending Commit

- Repository: `UIE3`
- Branch: `main`
- Starting commit: `4b7df596dae84461f39d5bb6557539aa9e31529f`
- Ending commit: `pending`（本阶段未创建 Git commit）
- Upstream NAFNet commit: `2b4af71ebe098a92a75910c233a3965a3e93ede4`

## Changed Files

1. `tools/train_baseline.py`
2. `tools/evaluate_baseline.py`
3. `tests/test_train_step.py`
4. `reports/PHASE_B2B_R1_REPORT.md`

## Assumptions

- 训练 smoke 模式严格要求同时提供 train 和 validation override，与 Prompt 明确要求一致。
- 评价 smoke 模式未被要求必须提供 override；未提供时仍使用 canonical validation manifest，但运行仍明确标记为 `SMOKE_TEST`、`formal_experiment=false`。
- Override 路径按当前 CLI 工作目录解析为绝对路径；解析后的路径、大小写不敏感的 `test.tsv` 文件名、符号链接及与 canonical test manifest 相同的文件对象均受 test 禁止规则约束。

## Implementation Summary

### Training CLI

- 新增 `--smoke-test`、`--train-manifest-override`、`--validation-manifest-override`。
- 未启用 smoke 时，任一 override 都立即报错；YAML 中的正式 manifest 校验仍严格要求：
  - `splits/lsui19/train.tsv`
  - `splits/lsui19/validation.tsv`
- Smoke 模式要求正整数 `--max-steps`、两个 override 同时存在、两个目标均为普通文件，并拒绝任何 `test.tsv` 目标。
- Manifest 选择只改变当前运行实际路径，不改写 YAML 或 canonical manifests。
- 正式运行输出：
  - `RUN_MODE=FORMAL`
  - `FORMAL_EXPERIMENT=true`
- Smoke 运行输出：
  - `RUN_MODE=SMOKE_TEST`
  - `FORMAL_EXPERIMENT=false`
  - 两个解析后的 override 实际路径。
- Smoke checkpoint config 增加：
  - `run_mode: smoke_test`
  - `formal_experiment: false`
  - `actual_train_manifest`
  - `actual_validation_manifest`
- 正式 checkpoint config 不添加 smoke provenance 字段，保留正式恢复配置语义。

### Evaluation CLI

- 新增 `--smoke-test`、`--validation-manifest-override`。
- 正式 evaluation 拒绝 override，并固定使用 canonical validation manifest。
- Smoke evaluation 仅允许 `--split validation`；`--split test` 即使在 smoke 模式也明确拒绝。
- Validation override 使用与训练入口相同的 test manifest 防护。
- 输出和 JSON summary 记录 run mode、`formal_experiment` 和实际 validation manifest。

### Test Definitions

`tests/test_train_step.py` 新增或扩展覆盖：

- 正式配置拒绝非 canonical train/validation manifest；
- 未启用 smoke 时训练和评价均拒绝 override；
- 训练 smoke 缺少 `max_steps`、非正 `max_steps`、缺少任一 override 时拒绝；
- 缺失 override 文件拒绝；
- train/validation override 指向 `test.tsv` 时拒绝；
- 合法训练 smoke 双 override 被接受；
- Smoke checkpoint config 记录实际 manifest provenance；
- 合法 validation smoke override 被接受；
- Smoke evaluation 的 test manifest override 和 test split 均被拒绝；
- 运行 manifest 选择函数前后，正式 YAML 和三个 canonical manifests 字节保持不变。

测试仅使用临时 TSV 元数据与合成 tensor 定义，不访问真实 LSUI 图片，不执行长训练。

## Commands Actually Executed

```text
sed -n ... ORDER_STUDY_PROTOCOL.md
sed -n ... ORDER_STUDY_DECISIONS.md
sed -n ... ORDER_STUDY_STATE.yaml
sed -n ... tools/train_baseline.py
sed -n ... tools/evaluate_baseline.py
sed -n ... tests/test_train_step.py
sed -n ... reports/PHASE_B2B_REPORT.md
pwd
git branch --show-current
git rev-parse HEAD
git status --porcelain=v1 -uall
git -C ../NAFNet status --porcelain=v1
git -C ../NAFNet rev-parse HEAD
python3 --version
python3 <AST/top-level-import/py_compile/compileall static checks>
python3 -B tools/train_baseline.py --help
python3 -B tools/evaluate_baseline.py --help
rg <CLI guard, output marker, and test-definition checks>
git diff --check
git diff -- <three authorized implementation paths>
```

未执行 pytest、训练、评价、真实 manifest 数据加载或包安装。

## Actual Static Outputs

- Python: `3.9.6`
- torch probe: `missing`
- AST: `AST_PARSE_OK files=3`
- CLI top-level torch imports: `none`
- `py_compile`: `PY_COMPILE_OK files=3`
- `compileall`: `COMPILEALL_OK files=3`
- Training CLI help: success，包含三个新增 smoke 参数。
- Evaluation CLI help: success，包含两个新增 smoke 参数。
- Guard/output marker grep: required definitions found。
- Required test-name grep: required definitions found。
- `git diff --check`: success。
- pytest: not run。
- Real-data smoke test: not run。

## Acceptance Criteria

- 正式配置和运行路径保护未删除或弱化。
- 训练 smoke override 仅在显式 smoke、正整数 `max_steps` 和双文件 override 条件下可选择。
- Test manifest 与 test split 保持禁止。
- Smoke 运行与 checkpoint provenance 明确标记为非正式实验。
- 云端运行时测试已定义但尚未执行，因此本阶段不能标记为 `PASS`。

## Unresolved Risks

- 本地缺少 torch，新增及既有 pytest 均未实际运行。
- 尚未在云端使用真实 1–4 样本 manifests 执行训练或 validation smoke。
- 尚未运行验证 smoke checkpoint 中四个 provenance 字段的实际序列化结果。
- 尚未验证云端 CLI 完整错误输出、DataLoader 创建及 finite-step 终止行为。
- Phase B2c runtime validation 仍未完成，且本阶段未授权自动返回 B2c。

## Supported Conclusions

Baseline CLI 的 manifest 保护规则已完成静态修复：正式模式仍固定使用 canonical train/validation manifests；显式 smoke 训练受正整数 `max_steps`、双 override、文件存在性和 test 禁止规则约束；validation smoke override 同样保持 test 禁用。

## Unsupported Conclusions

- 尚未通过云端 pytest；
- 尚未执行真实 LSUI smoke test；
- 尚未证明 optimizer 已在真实数据上完成更新；
- 尚未验证真实 smoke checkpoint 的保存与恢复；
- 尚未完成小样本过拟合验证；
- 尚未完成正式 baseline 训练或评价；
- 尚未执行最终 test 评价；
- 尚无 UIE 性能或论文实验结论。

## Next Phase Authorization

`false`

不得自动返回 B2c 或进入任何后续阶段。

## Final Phase Status

IMPLEMENTED_NOT_RUNTIME_VALIDATED
