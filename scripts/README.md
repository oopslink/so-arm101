# 配套脚本

在项目根目录运行。Linux脚本使用已经验证可用的LeRobot 0.6.1环境。脚本不含凭据；`config/lab.env`仅放非秘密参数。

| 脚本 | 来源/用途 | 是否驱动硬件 |
|---|---|---|
| `collect-data.sh` | 历史采集脚本，改为从环境变量读取设备与目录；独立批次/展示/旧模式resume | 是 |
| `check-data.py` | 历史批次结构检查及每条中间帧解码 | 否 |
| `merge_batches.py` | 新整理，调用0.6.1聚合API，保留源文件 | 否 |
| `ensure-featurize-pi0-env.sh` | 历史依赖检查补齐脚本；`--check-only`不安装 | 否，会修改Python环境（除check-only） |
| `metrics_from_log.py` | 新整理，导出CSV，明确标记缩写step为近似 | 否 |
| `plot_metrics.py` | 新整理，CSV画五类指标 | 否 |
| `plot_hardware_trace.py` | 新整理，绘制历史JSONL；按文本日志划分范围、排除回位 | 否，仅读取磁盘日志 |
| `package_release.py` | 新整理，保留训练处理器、加入tokenizer和日志 | 否，新建输出目录 |
| `verify_release.py` | 新整理，实际反归一化测试与离线加载；仅适配六维绝对MEAN_STD动作 | 否 |
| `pi0_dataset_replay.py` | 历史数据帧离线预测；名字含replay但不控制机器人 | 否 |
| `init_safe_speed.py` | 历史同步目标和舵机速度初始化 | **是：关/开扭矩，可能运动或下坠** |
| `pi0_trace_run.py` | 历史rollout诊断包装，保留真实send_action | **是：不是dry run** |

新脚本完成语法及不依赖硬件的测试；Spark开机后已核对实际聚合API参数与部署脚本哈希，但没有运行新采集/训练/GPU集成测试。依赖内部API的脚本升级LeRobot后应先离线检查，不应直接试机器人。

本地自检：`python3 scripts/test_docs_tools.py`，覆盖语法、日志step精度/续训顺序、空后处理配置拒绝。此次还用真实V3日志重建了3000条指标，并实际运行画图脚本生成图表。

## 最常用命令

```bash
# [Spark]
source config/lab.env
sh scripts/collect-data.sh --batches 8 --episodes-per-batch 10
python scripts/check-data.py 1-8 --base-root "$BASE_DATASET_ROOT" --repo-prefix "$BASE_REPO_ID"
python scripts/merge_batches.py --base-root "$BASE_DATASET_ROOT" --repo-prefix "$BASE_REPO_ID" \
  --batches 8 --output "$MERGED_ROOT" --repo-id "$DATASET_REPO_ID"
```

不完整批次不会自动删除；`--batches`是最后批次编号。每批不是10条时，检查脚本加`--expected M`。

```bash
# [Mac；专用画图环境，不改机器人环境]
python3 -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install matplotlib numpy
python scripts/plot_metrics.py evidence/v3-training-metrics.csv reports/v3-metrics.png
```

对新训练：`python scripts/metrics_from_log.py /path/training.log /path/metrics.csv`。CSV文件可以随时重建；不要覆盖唯一原始日志。图中的logger显存不等于整卡实时占用，step有缩写时横轴会保留相应精度限制。

## 为什么没有附“通用一键修模型”

错误的postprocessor需要恢复对应训练统计，而不是从本仓库复制一份固定JSON给所有模型。`evidence/v3-postprocessor-fixed.json`仅作本次证据。

同样，旧future4重标注脚本存在跨文件episode边界/统计更新局限，没有作为通用脚本发放。先使用原始action走通教程；要重标注，应保留原数据并建立单元测试后再训练。
