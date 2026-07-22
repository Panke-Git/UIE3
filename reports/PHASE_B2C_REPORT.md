# Phase B2c Report — Baseline Runtime and Real-Data Smoke Validation

## Phase Objective

依据已提交的云端运行证据，归档 NAFNet-small baseline 基础设施的单元与集成测试、真实 LSUI smoke 训练、checkpoint 保存与恢复、validation 逐图指标导出和 final-test split 保护结果。本归档步骤不修改代码或实验资产，也不重新运行任何实验。

## Governance Confirmation

- `current_phase.id`: `B2c`
- `current_phase.name`: `baseline_runtime_and_real_data_smoke_validation`
- `current_phase.authorized`: `true`
- `current_phase.next_phase_authorized`: `false`
- `reports/PHASE_B2C_REPORT.md` 同时位于 `phase_permissions.writable_paths` 和 `phase_outputs.required`。
- 本步骤唯一创建文件为 `reports/PHASE_B2C_REPORT.md`。

## Git and Evidence Provenance

- Branch: `main`
- 环境快照记录的 Git commit: `4dbbd1acfc2314d5d4ab9bf78686f4ecaf7e6598`
- 首次训练和 resume 日志打印的实际运行 commit: `bd8ee664beb75616dc7ec6f273afadb0a060f6d1`
- Runtime evidence commit: `0a6bcd1`（新增八项 Phase B2c runtime artifacts）
- 本次报告归档起始 commit: `796b3e527bb294ccf3fb880b3669d09489110ed6`
- Ending commit: `pending`（本步骤未创建 Git commit）

环境快照产生于 smoke-manifest remediation 之前，训练日志产生于 remediation commit 之后，因此报告分别保留两项 commit，不将其错误合并为同一时间点。

## Changed Files

- `reports/PHASE_B2C_REPORT.md`

未修改 Python、tests、configs、formal manifests、治理文件或 runtime evidence。

## Cloud Environment

依据 `reports/runtime/phase_b2c/environment.txt`：

- Platform: `Linux-5.15.0-139-generic-x86_64-with-glibc2.35`
- Python: `3.12.13`（Anaconda build）
- Python executable: `/root/miniconda3/envs/pax/bin/python`
- PyTorch: `2.13.0+cu126`
- CUDA runtime: `12.6`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 3090`
- cuDNN: `91002`
- NumPy: `2.4.4`
- Pillow: `12.2.0`
- PyYAML: `6.0.3`
- pytest: `9.1.1`
- Environment-file Git commit: `4dbbd1acfc2314d5d4ab9bf78686f4ecaf7e6598`
- Dataset root: `/root/autodl-tmp/pro/publicdata/LSUI19_dup_train`

## Pytest Results

依据 `pytest.log` 和 `pytest_exit_code.txt`：

- 实际测试数量: `30`
- Passed: `30`
- Failed: `0`
- Warnings: `161`
- Duration: `6.56s`
- `pytest_exit_code`: `0`

Warning 构成：

- 1 条 `requires_grad=True` tensor 转 scalar 的 `UserWarning`；
- 官方 NAFNet `ctx.saved_variables` deprecated warning 32 条；
- UIE3 迁移实现的同类 deprecated warning 共 128 条，其中 NAFNet import tests 32 条、train-step tests 96 条。

这些 warning 未导致测试失败；`ctx.saved_variables` 属于后续 PyTorch 兼容性维护事项。

## Real LSUI Smoke Test

### Run Identity and Data

- `RUN_MODE=SMOKE_TEST`
- `FORMAL_EXPERIMENT=false`
- Train override: `/tmp/UIE3_phase_b2c/train_4.tsv`
- Validation override: `/tmp/UIE3_phase_b2c/validation_4.tsv`
- 两个 runtime manifest 均按运行证据标识为 4 样本 manifest。
- Validation 日志和 CSV 独立确认 validation 样本数为 4。
- Dataset root: `/root/autodl-tmp/pro/publicdata/LSUI19_dup_train`
- CSV 中的实际样本路径来自 `Train/input/*.jpg` 和 `Train/GT/*.jpg`，表明 smoke 使用真实 LSUI 配对图像，而非合成图片。
- 模型参数量: `2,846,755`
- Seed: `3407`

### CUDA, Forward, Backward, Optimizer and AMP

- Device 日志为 `cuda`，云端环境确认 CUDA available。
- 训练产生连续、有限的 loss 和 validation metric 记录并将 `global_step` 推进至 40，支持 CUDA forward、backward 和 optimizer step 已成功完成。
- 对首次训练和 resume 的 40 条 JSON 训练记录进行只读复核，所有 `train_loss`、validation PSNR、validation SSIM、best PSNR 和 learning rate 均为有限值。
- 日志未出现 NaN、Inf、shape mismatch 或运行异常。
- 实际日志中的 AMP 配置为 `training.amp: true`，运行 device 为 `cuda`。日志未单独打印内部 `amp_enabled` 或 scaler 数值，因此不对日志未记录的 AMP 内部状态作额外推测。

## Initial Training

- Runtime max steps: `20`
- 进入训练循环前的 `global_step` 未在日志中单独打印。
- 第一条实际训练记录: `global_step=1`
- 最后一条实际训练记录: `global_step=20`
- 共记录 20 个连续训练步，无缺号。
- 结束时 validation PSNR: `15.547589778900146`
- 结束时 validation SSIM: `0.6396191418170929`
- 初次运行的最佳 validation PSNR: `15.639548778533936`
- 日志多次记录 `is_best=true`，证明 best-checkpoint 分支被触发；日志未提供独立目录清单，因此不额外声称 `best.pt` 的文件存在性检查结果。
- 配置记录 `save_every: 1`，但日志未逐个列出 epoch checkpoint 文件。
- `last.pt` 的成功保存由后续 resume 从 `/tmp/UIE3_phase_b2c/overfit_run/last.pt` 成功加载直接验证。

## Checkpoint Resume

依据 `train_resume.log`：

- Resume checkpoint: `/tmp/UIE3_phase_b2c/overfit_run/last.pt`
- 恢复前 `global_step`: `20`
- Resume 后第一条训练记录: `global_step=21`
- Resume 后最终 `global_step`: `40`
- Expected minimum step: `40`
- Final epoch: `39`
- `model_state_present`: `True`
- `optimizer_state_present`: `True`
- `resume_exit_code`: `0`
- `Checkpoint resume verification`: `PASS`

Resume 记录从 21 连续增长到 40，与恢复前的 20 无断点，支持 model/optimizer state 严格恢复和 `global_step` 连续性验收。

## Validation Export

依据 `validation.log` 和 `validation_metrics.csv`：

- `RUN_MODE=SMOKE_TEST`
- `FORMAL_EXPERIMENT=false`
- Split: `validation`
- Actual validation manifest: `/tmp/UIE3_phase_b2c/validation_4.tsv`
- Validation 样本数: `4`
- CSV 物理行数: `5`（1 行 header + 4 行数据）
- CSV 数据行数: `4`
- CSV 列名:
  - `sample_id`
  - `input_relative_path`
  - `gt_relative_path`
  - `psnr_rgb`
  - `ssim_rgb`
- 每张图像均包含 `psnr_rgb` 和 `ssim_rgb`。
- 对 CSV 进行只读解析，全部 8 个指标值均为有限值。
- Mean RGB PSNR: `16.11678147315979`
- Mean RGB SSIM: `0.6814638525247574`
- `validation_exit_code`: `0`

Validation 在 resume 运行完成后归档，并与 resume 最终步的 validation 指标对应；但 `validation.log` 本身未回显 `--checkpoint` 参数路径，因此仅将 `/tmp/UIE3_phase_b2c/overfit_run/last.pt` 作为 resume 日志明确确认的恢复 checkpoint，不虚构 validation 日志中不存在的 checkpoint 行。

## Test Split Guard

依据 `test_split_guard.log` 和 runtime 目录只读检查：

- `--split test` 被明确拒绝。
- Guard error: `Phase B2b-r1 does not authorize final test evaluation; --split must be validation even in smoke-test mode.`
- `test_guard_exit_code`: `2`（非 0）
- `forbidden_test.csv`: 未生成。
- 日志没有任何 final-test 指标输出，final test 未被实际计算。

## Non-Blocking Issue

- Test guard 提示文字仍引用 `Phase B2b-r1`。
- 实际保护行为正确：test split 被拒绝、退出码为 2、没有生成 `forbidden_test.csv`。
- 该问题仅属于提示文本维护问题，不影响当前验收。

## Acceptance Criteria

| Criterion | Status | Runtime evidence |
|---|---|---|
| pytest 通过 | PASS | 30 passed, 0 failed, 161 warnings, exit code 0 |
| 真实 LSUI 读取成功 | PASS | 真实 dataset root、4-sample manifests、真实 Train input/GT 路径及成功训练/validation |
| CUDA forward/backward 成功 | PASS | Device=cuda；40 个有限训练步成功完成 |
| Optimizer 更新成功 | PASS | global_step 连续推进 1→20、resume 后 21→40，loss 为有限值 |
| Checkpoint 保存成功 | PASS | `last.pt` 被后续 resume 成功加载 |
| Checkpoint strict resume 成功 | PASS | model/optimizer state present，resume verification PASS，exit code 0 |
| global_step 连续 | PASS | old=20，resume 后 21→40，无断点 |
| Validation 逐图 CSV 成功 | PASS | 4 个样本、4 个数据行、所需 5 列、exit code 0 |
| PSNR 和 SSIM 有限 | PASS | CSV 的 4 个 PSNR 与 4 个 SSIM 均通过 finite 检查 |
| Test split 被拒绝 | PASS | 明确 ValueError，guard exit code 2，未生成 forbidden CSV |
| 未修改源码、正式 manifest 或数据集 | PASS | runtime commit 相对 remediation commit 只新增八项 evidence artifacts；本归档前 worktree clean；日志仅记录数据读取路径，无数据改写证据 |

所有验收条件均有现有证据支持。

## Commands Executed During Report-Only Closure

仅执行只读命令与本报告写入：

```text
wc -l <required evidence file>
sed -n '1,10000p' <required evidence file>
git branch --show-current
git rev-parse HEAD
git status --porcelain=v1 -uall
git -C ../NAFNet status --porcelain=v1
git log --oneline --decorate -8
git show --no-patch <recorded commits>
git diff --name-only <runtime commit ranges>
python3 <standard-library-only read-only parsing of logs and validation CSV>
rg --files reports/runtime/phase_b2c | rg 'forbidden_test\.csv'
```

未重新运行 pytest、训练或评价；未安装依赖；未写入 runtime artifacts。

## Unresolved Risks

- 本阶段是 4 样本、40 optimizer steps 的 smoke validation，不代表正式训练规模。
- Validation 仅覆盖 4 样本；不能作为完整 validation 指标。
- Validation 日志未回显 checkpoint CLI 参数路径，尽管其运行顺序和数值与 resume 后最终记录一致。
- AMP 配置和 CUDA device 已记录，但日志未单独输出内部 scaler/`amp_enabled` 状态。
- Test guard 的阶段名称提示需要未来获授权时维护；保护行为本身正确。

## Supported Conclusion

“NAFNet-small baseline基础设施已在云端通过单元与集成测试、真实LSUI数据加载、CUDA forward/backward、optimizer更新、checkpoint保存与恢复、validation逐图指标导出和test-split保护验证。”

## Unsupported Conclusions

- 尚未进行正式baseline训练；
- 尚未在完整validation上获得正式指标；
- 尚未评估final test；
- 尚未完成超参数选择；
- 尚未验证多随机种子稳定性；
- 尚未实现颜色算子；
- 尚未实现散射算子；
- 尚未验证算子顺序假设；
- 尚无论文主结果。

## Next Phase Authorization

`false`

不得自动进入下一阶段。

## Final Phase Status

PASS
