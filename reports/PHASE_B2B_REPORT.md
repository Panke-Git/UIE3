# Phase B2b Report — Baseline Training Infrastructure Static Implementation

## Phase Objective

本阶段仅静态实现单一路径 `input -> NAFNet-small -> restored image` 的 baseline 数据、Charbonnier 损失、RGB PSNR/SSIM 指标、随机种子、checkpoint、训练、validation 评价、配置和云端运行时测试基础设施。不执行训练、真实数据读取、pytest 或最终 test 评价。

## Governance Confirmation

- `current_phase.id`: `B2b`
- `current_phase.name`: `baseline_training_infrastructure_static_implementation`
- `current_phase.authorized`: `true`
- `current_phase.next_phase_authorized`: `false`
- 本报告及本阶段全部创建路径均在 `phase_permissions.writable_paths` 中，并列于 `phase_outputs.required`。
- 未修改治理文件、`third_party/nafnet/**`、`src/models/backbones/nafnet_small.py`、`splits/lsui19/**` 或数据集图像。

## Starting Commit

- Repository: `UIE3`
- Branch: `main`
- Starting commit: `05f5b4331f8f37e4f17d17def5cc2bc2cd5543bd`
- Ending commit: `pending`（本阶段未创建 Git commit）

## Changed Files

本阶段创建以下 20 个授权文件：

1. `requirements/baseline.txt`
2. `configs/nafnet_small_lsui.yaml`
3. `src/data/__init__.py`
4. `src/data/paired_image_dataset.py`
5. `src/losses/__init__.py`
6. `src/losses/charbonnier.py`
7. `src/metrics/__init__.py`
8. `src/metrics/image_metrics.py`
9. `src/utils/__init__.py`
10. `src/utils/seed.py`
11. `src/engine/__init__.py`
12. `src/engine/checkpoint.py`
13. `src/engine/trainer.py`
14. `tools/train_baseline.py`
15. `tools/evaluate_baseline.py`
16. `tests/test_manifest_dataset.py`
17. `tests/test_loss_metrics.py`
18. `tests/test_checkpoint.py`
19. `tests/test_train_step.py`
20. `reports/PHASE_B2B_REPORT.md`

## Implementation Summary

- `requirements/baseline.txt` 只声明 NumPy、Pillow、PyYAML 和 pytest；明确要求用户根据云端 CUDA 环境另行安装 PyTorch，未固定 torch。
- Manifest 数据集严格解析三列 TSV，拒绝空或重复 `sample_id`、非法列数、路径逃逸、缺失文件及 input/GT 尺寸不一致；Pillow 显式转 RGB，并输出 `[3,H,W]`、`float32`、`[0,1]` tensor。
- 训练时 crop、reflect padding、水平/垂直翻转和 90 度旋转均对 input/GT 使用同一空间参数；非训练模式不执行随机 crop 或增强。
- Charbonnier loss 实现为 `sqrt((prediction-target)^2 + epsilon^2)`，包含 shape、浮点类型和有限值检查，不含辅助损失。
- RGB PSNR/SSIM 按逐图计算后再汇总；仅 prediction 在指标计算前 clamp，target 必须已在 `[0,1]`。SSIM 固定使用 11×11、sigma 1.5 的 Gaussian window，并按 RGB 通道独立计算后平均。
- 随机种子覆盖 Python、NumPy、torch CPU 和可用时的 torch CUDA；确定性行为由配置控制，不无条件启用确定性算法。
- Checkpoint 包含必需 schema、模型/优化器/可选 scheduler/scaler 状态、epoch、global step、最佳 validation PSNR、配置、随机种子、Git commit 和 UTC 时间戳；采用同目录临时文件与原子替换，支持 CPU map location 和 strict model load。
- Trainer 支持 CPU/CUDA、仅 CUDA AMP、单步训练、validation epoch、有限值与梯度检查、可选梯度裁剪、最佳 validation PSNR、保存及严格恢复；训练 loss 前不 clamp 输出。
- 训练入口只解析正式 train/validation manifest，强制唯一 NAFNet-small 配置和 AdamW/Charbonnier 语义，支持 `max_steps`、恢复、非覆盖输出目录、参数量/配置/Git commit 输出和 JSONL 日志。
- 评价入口只允许 `--split validation`，明确拒绝 test，输出逐图 CSV 和 validation 数据集平均 PSNR/SSIM。
- 配置值为 baseline smoke-test 初始值，未将其描述为已优化超参数。
- 四个 pytest 文件仅使用临时目录或合成 tensor，未访问真实 LSUI 数据，也未定义多轮训练。

## Commands Actually Executed

以下命令仅进行静态检查或环境查询；未执行 pytest、训练、评价或真实数据访问：

```text
python3 --version
python3 <importlib.util.find_spec dependency probe>
python3 <PyYAML import probe>
python3 <AST parse of 17 Phase B2b Python files>
python3 <py_compile of 17 Phase B2b Python files using a temporary pycache prefix>
python3 <compileall of Phase B2b source/tool/test paths using a temporary pycache prefix>
python3 -B tools/train_baseline.py --help
python3 -B tools/evaluate_baseline.py --help
rg <forbidden dependency import patterns> <Phase B2b paths>
rg <out-of-scope operator/routing patterns> <Phase B2b paths>
git diff --check
python3 <final 20-path authorization and text-format audit>
git diff --quiet
git diff --cached --quiet
git -C ../NAFNet status --porcelain
git branch --show-current
git rev-parse HEAD
git status --porcelain=v1 -uall
```

YAML 运行时语法解析仅在 PyYAML 可用时获准执行；本机缺少 PyYAML，因此未执行 `yaml.safe_load`。

## Actual Outputs

- Python: `3.9.6`
- AST: `AST_PARSE_OK files=17`
- `py_compile`: `PY_COMPILE_OK files=17`
- `compileall`: `COMPILEALL_OK`
- Training CLI: `TRAIN_CLI_HELP_OK`
- Evaluation CLI: `EVALUATE_CLI_HELP_OK`
- 禁用依赖 import grep: `FORBIDDEN_DEPENDENCY_IMPORTS_OK`
- 越界 operator/routing grep: `OUT_OF_SCOPE_MODEL_TERMS_OK`
- `git diff --check`: `GIT_DIFF_CHECK_OK`
- `yaml` import: `ModuleNotFoundError: No module named 'yaml'`
- Git branch/HEAD: `main` / `05f5b4331f8f37e4f17d17def5cc2bc2cd5543bd`
- 最终写入范围：`WRITE_SCOPE_OK files=20`；20 个路径全部为本阶段授权输出。
- 已跟踪文件：`TRACKED_FILES_UNCHANGED` / `UIE3_TRACKED_DIFFS_NONE`
- 文本格式：`TEXT_FORMAT_OK`
- 上游工作树：`NAFNET_WORKTREE_UNMODIFIED`
- 未产生训练日志、checkpoint、评价 CSV、预览图或真实数据输出。

## Tests Defined but Not Run

### `tests/test_manifest_dataset.py`

- 正常 manifest、RGB 转换、tensor shape/dtype/range；
- 重复 ID、非法列数、路径逃逸、尺寸不匹配拒绝；
- paired crop/augmentation；
- 小图 paired reflect padding 与 `pad_if_smaller=false` 拒绝；
- validation 不使用随机 crop/增强。

### `tests/test_loss_metrics.py`

- Charbonnier 已知值、有限 backward、shape mismatch；
- PSNR 已知 MSE、相同图像 `+inf` 行为；
- 相同图像 SSIM、逐图 batch 返回值、过小图像拒绝。

### `tests/test_checkpoint.py`

- 原子保存及无临时残留；
- model/optimizer/counter round trip；
- 必需字段缺失拒绝；
- strict model load。

### `tests/test_train_step.py`

- 合成数据上的 NAFNet-small 单步 forward/backward；
- loss/梯度有限及参数更新；
- CPU 自动禁用 AMP；
- validation 逐图 PSNR/SSIM；
- checkpoint resume 后 global step 连续。

上述测试仅完成定义，本地未执行。

## Dependency Status

本机静态探测结果：

- `torch`: missing
- `numpy`: missing
- `PIL`: missing
- `yaml`: missing
- `pytest`: missing

未安装任何依赖。由于 torch 不可用，严格遵循阶段要求，不运行 pytest。两个 CLI 将运行期依赖延迟到参数解析之后，因此 `--help` 无需导入 torch 即可成功。

## Acceptance Criteria

- 授权的 20 个必需输出均已创建。
- Python 静态解析和编译检查通过。
- 两个 CLI 帮助接口在无 torch 环境通过。
- 未发现禁止依赖 import 或越界算子、顺序、路由实现。
- 未读取 test manifest、未进行最终 test 评价。
- 未执行运行时测试，因此本阶段不具备 `PASS` 条件。

## Unresolved Risks

- 当前环境无 torch、NumPy、Pillow、PyYAML 或 pytest，所有运行时 import、tensor 运算、Pillow 解码及 YAML 加载仍待云端验证。
- 真实 LSUI manifest、图像解码、worker 行为、成对 crop/增强和小图 reflect padding仍待云端 pytest 与 smoke test 验证。
- NAFNet-small 的真实 optimizer update、显存占用、AMP 路径和 validation 指标吞吐尚未验证。
- Checkpoint 在目标云端 PyTorch 版本中的完整保存、恢复和跨设备加载尚未验证。
- SSIM 数值约定虽已固定实现，仍需运行时测试确认目标 PyTorch 算子路径下的数值行为。
- `max_steps`、恢复训练、输出目录保护及最佳 checkpoint 选择仍待 B2c 授权后的受控运行验证。

## Supported Conclusions

“NAFNet-small baseline的数据、损失、指标、checkpoint、训练和validation评价基础设施已经完成静态实现，并定义了云端运行时测试。”

## Unsupported Conclusions

- 尚未通过云端pytest；
- 尚未读取真实LSUI batch；
- 尚未完成optimizer真实数据更新；
- 尚未验证checkpoint真实恢复；
- 尚未完成小样本过拟合；
- 尚无正式baseline指标；
- 尚未实现颜色算子；
- 尚未实现散射算子；
- 尚未验证顺序假设。

## Next Phase Authorization

`false`

未授权自动进入 B2c。

## Final Phase Status

IMPLEMENTED_NOT_RUNTIME_VALIDATED
