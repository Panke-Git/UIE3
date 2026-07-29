# Phase B3b-4 三随机种子正式 Validation 聚合报告

## 结论

Phase B3b-4 的全部验收条件均已满足。三个预先指定随机种子 3407、1234、2027 的正式 validation 结果为：

- PSNR RGB：`27.2312146199214 ± 0.16617570971014672` dB；
- SSIM RGB：`0.8927656310461302 ± 0.0015307680208464887`。

这里的主要变异性统计均为**样本标准差**，分母为 `n-1`，即 `ddof=1`。适合论文表格的建议舍入格式为：PSNR `27.231 ± 0.166` dB，SSIM `0.8928 ± 0.0015`。

本报告只聚合既有正式运行证据，没有运行训练、没有加载 checkpoint 进行评价、没有选择或排除 seed，也没有评估 final test。

## 阶段授权与证据边界

- `current_phase.id == B3b-4`；
- `current_phase.name == baseline_three_seed_validation_aggregation`；
- `current_phase.authorized == true`；
- `current_phase.next_phase_authorized == false`；
- 聚合时工作区 Git HEAD：`0fd87f1a7967b02c12d987f55513b015aa5309d4`；
- seed 3407、1234、2027 的正式报告均为 `PASS`。

机器可读记录位于：

- `reports/runtime/phase_b3b_4/three_seed_validation_metrics.csv`；
- `reports/runtime/phase_b3b_4/three_seed_validation_summary.json`。

## 三个 seed 的逐项结果

每行均来自该 seed 的独立 `best_validation_metrics.csv` 对完整 385 张正式 validation 样本的重新聚合。

| seed | Mean PSNR RGB | Mean SSIM RGB | validation 样本数 | best epoch | best global_step | last epoch | last global_step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3407 | 27.319515354602366 | 0.8942945567044345 | 385 | 192 | 167267 | 199 | 173334 |
| 1234 | 27.334599338878284 | 0.8927693091429674 | 385 | 180 | 156865 | 199 | 173331 |
| 2027 | 27.039529166283547 | 0.8912330272909883 | 385 | 188 | 163799 | 199 | 173334 |

| seed | checkpoint 选择 PSNR | checkpoint/CSV PSNR 绝对差 | train exit | validation exit | final test |
|---:|---:|---:|---:|---:|:---:|
| 3407 | 27.319525305017248 | 9.950414881387815e-06 | 0 | 0 | 未评估 |
| 1234 | 27.334571999388857 | 2.7339489427902208e-05 | 0 | 0 | 未评估 |
| 2027 | 27.03952445736179 | 4.708921757412554e-06 | 0 | 0 | 未评估 |

三个 checkpoint/CSV PSNR 差值均在 `1e-4` dB 归档容差内。checkpoint 选择规则对三个 seed 一致，均为 `highest formal-validation RGB PSNR`。

## Checkpoint 与配置来源

| seed | best.pt SHA256 | config SHA256 | best checkpoint Git commit |
|---:|:---|:---|:---|
| 3407 | `81d15252045ff96bf3c241a580ec35076eea035b7bdafe9c8cb1393c827fdd81` | `ee812ff2c6621a9e01691e016b497eaf94af152320d141b80d69209abd917e80` | `79682aafff31fec602921acc63ad07faa9710938` |
| 1234 | `f45d08e0fd3854c9d7c71a635a469af7d32f3207cc558ebbe065e52da2efc667` | `5b1dde02deec9984d17760224be975bed3b1a45d25f5b399e85fb27ec0c1e396` | `0a713111f34a3bccacecbadd6131bd641b357c7f` |
| 2027 | `29814cd2c0ab7e893d0bd329ded2c9875840ab8c51d0163e412a63297023c0f0` | `e82e17d95822da8b64e5439c3bdf8bc75defd015fa1c06588c26e4ad8fc024dc` | `f0a802e2bbd82ed8621c8f01087d36d8c09d680e` |

三个 seed 的 `best.pt` 和 `last.pt` 均有效，且 model、optimizer、GradScaler 状态完整。三次训练均从 epoch 0 开始并完成 200 epochs，即 epoch 0–199；每次均使用各自 `best.pt` 对完整 385 张正式 validation 数据进行评价。

## 配置一致性

三份 YAML 配置经解析后进行了递归逐字段比较。比较前只移除了：

- `experiment.name`；
- `experiment.seed`。

移除后，三份配置完全一致；两两原始配置的差异路径也严格只有上述两个字段。三个配置文件的 SHA256 不同是预期现象，不构成协议不一致。

显式核对通过的固定协议包括：

- NAFNet-small：`img_channel=3`、`width=32`、`enc_blk_nums=[2,2,2]`、`middle_blk_num=4`、`dec_blk_nums=[2,2,2]`；
- train manifest：`splits/lsui19/train.tsv`；
- validation manifest：`splits/lsui19/validation.tsv`；
- `patch_size=256`、`batch_size=4`、`num_workers=4`；
- augmentation：`hflip=true`、`vflip=true`、`rot90=true`、`pad_if_smaller=true`；
- Charbonnier loss：`epsilon=1e-3`；
- AdamW：`learning_rate=2e-4`、`weight_decay=0`、`betas=[0.9,0.999]`；
- `AMP=true`、`epochs=200`、`validate_every=1`、`save_every=10`；
- metrics 配置完全一致；
- 不含 test manifest；
- checkpoint 选择规则一致：最高正式 validation RGB PSNR。

## 逐图 CSV 一致性与均值复算

三份 CSV 均恰好包含 385 行数据，且：

- `sample_id` 集合完全一致；
- `sample_id -> input_relative_path` 映射完全一致；
- `sample_id -> gt_relative_path` 映射完全一致；
- 每份 CSV 的重复 `sample_id` 数均为 0；
- 全部 1,155 个 PSNR 和 1,155 个 SSIM 值均为有限数。

使用 `math.fsum` 对每份 CSV 分别重新计算均值，并以绝对容差 `1e-10` 与对应 `run_summary.json` 核对：

| seed | PSNR 复算绝对差 | SSIM 复算绝对差 | 结果 |
|---:|---:|---:|:---:|
| 3407 | 0.0 | 0.0 | PASS |
| 1234 | 0.0 | 0.0 | PASS |
| 2027 | 0.0 | 0.0 | PASS |

因此，三份 CSV 的样本与路径映射完全对齐，所有逐图指标有限，且 CSV 复算均值与对应 run summary 完全一致。

## 三 seed 聚合统计

聚合对象是三个 seed 级别的 `mean_psnr_rgb` 和 `mean_ssim_rgb`。算术均值使用 Python 标准库 `statistics.fmean`；主要标准差使用 `statistics.stdev`，即样本标准差、分母 `n-1`、`ddof=1`；附带的总体标准差使用 `statistics.pstdev`，即分母 `n`、`ddof=0`。

| 指标 | 算术均值 | 样本标准差 ddof=1（主要） | 总体标准差 ddof=0 | 最小值 | 最大值 | 极差 |
|:---|---:|---:|---:|---:|---:|---:|
| PSNR RGB | 27.2312146199214 | 0.16617570971014672 | 0.1356818988115731 | 27.039529166283547 | 27.334599338878284 | 0.29507017259473756 |
| SSIM RGB | 0.8927656310461302 | 0.0015307680208464887 | 0.00124986685521466 | 0.8912330272909883 | 0.8942945567044345 | 0.0030615294134461735 |

总体标准差仅作为补充记录，不替代主要报告的样本标准差。

### 论文表格建议舍入

| 指标 | 三 seed Mean ± Sample Standard Deviation |
|:---|:---|
| PSNR RGB（3 位小数） | `27.231 ± 0.166` dB |
| SSIM RGB（4 位小数） | `0.8928 ± 0.0015` |

## Acceptance Criteria

| 验收项 | 状态 |
|:---|:---:|
| 三个正式 seed 报告均为 PASS | PASS |
| 三个 run_summary 均有效 | PASS |
| 配置除 name 和 seed 外完全一致 | PASS |
| 三份 CSV 均为 385 行 | PASS |
| 三份 CSV 样本与路径映射完全一致 | PASS |
| 所有指标有限 | PASS |
| CSV 复算值与 run_summary 在绝对容差 `1e-10` 内一致 | PASS |
| 三 seed 聚合统计计算成功 | PASS |
| 不存在 seed 选择或排除 | PASS |
| final test 未评估 | PASS |

## Supported Conclusion

“NAFNet-small baseline已在固定协议下完成三个预先指定随机种子 3407、1234和2027的正式训练与完整validation评价。三seed结果可使用均值±样本标准差（ddof=1）报告。”

## Unsupported Conclusions

- 尚未获得 final test 结果；
- validation 聚合不是 final test 结果；
- 不能从三个 seed 中择优选择一个作为最终 baseline；
- 尚未验证颜色与散射逆算子；
- 尚未验证算子非交换性；
- 尚未验证 Oracle gap；
- 尚未验证自适应 router。

**Final Phase Status: PASS**

**Next phase authorization: false**
