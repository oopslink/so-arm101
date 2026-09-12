# 证据、来源与复现边界

路径名称沿用知识总结中的约定：`<ROBOT_LEROBOT_ROOT>`、`<ROBOT_DATA_ROOT>`、`<ROBOT_MODEL_ROOT>`、`<CLOUD_WORK_ROOT>`、`<CLOUD_DATA_ROOT>`和`<MAC_MODEL_ROOT>`分别代表对应机器上的目录；`<ORIGINAL_PROJECT_ROOT>`表示整理本文前的原始代码项目。为避免泄露或固化个人目录，历史配置与日志摘录也使用这些名称，不再保留机器上的绝对路径。

## 本次实际读取的材料

- 原项目 `<ORIGINAL_PROJECT_ROOT>`：采集、批次校验、动作追踪、离线预测、舵机速度初始化、云端依赖检查、打包代码、历史截图。
- Mac发布副本 `<MAC_MODEL_ROOT>/qiaomuyu_search_dual_pi0_v3_follower_future4/final_pretrained`：训练配置、完整训练日志、指标CSV、图表、前后处理配置、统计文件头。
- 实验过程记录：现场观察、试验目标变化、报错堆栈及运行结果。
- LeRobot官方版本源码/文档：用于核对SO101、PI0、RTC、聚合API与配置字段。

[只读核验](spark-readonly-audit.md)记录了配置、版本、数据数值、历史60秒日志和两张历史相机快照。核验过程没有启动训练或机器人，没有打开摄像头或加载策略执行，也没有把模型权重和原始训练数据复制进文档项目。

## 仓库中的证据文件

| 文件 | 含义 |
|---|---|
| `../docs/assets/demo.mp4` | 本实验留存的成品演示视频；对应训练版本、运行参数和是否剪辑未另行核验 |
| `v3-train-config.json` | Mac旧发布副本中的历史配置，保留V2命名残留；不可直接当新配置执行 |
| `v3-training-metrics.csv` | 3000条历史日志提取指标；step是旧提取程序按每10步恢复的计数 |
| `v3-postprocessor-fixed.json` | 原项目留存的V3专用修复配置；必须与对应统计匹配 |
| `../docs/assets/v3-training-metrics.png` | 历史真实CSV绘制的图表，51点平滑 |
| `../docs/assets/06-merged-dataset-80.png` | 早期合并完成，含TorchCodec失败/PyAV回退 |
| `../docs/assets/09-training-completed-checkpoints.jpg` | V1 3000步完成及目录体积，不是V3 |
| `../docs/assets/14-wandb-v2-running.png` | V2早期运行，不是最终30000步曲线 |
| `../docs/assets/workflow.png`、`traditional-programming-pipeline.png`、`imitation-learning-pipeline.png`、`postprocessor.png`等十四张解释图 | 使用本实验场景图作为风格参考，由内置图像生成工具逐张生成或编辑；其中十二张含整机/任务物体的插图又按本机实拍修正结构。主题提示词与边界见`../docs/assets/README.md`，不是实测画面 |
| `spark-readonly-audit.md` | 机器人工作站只读核验：环境、补丁、数据、模型配置哈希、历史日志 |
| `spark/`中的60秒log/JSONL | Spark上2026-09-10历史记录，不是本次新运行 |
| `../docs/assets/spark-historical-60s.png` | 根据上述稀疏JSONL绘制，已排除回位阶段 |
| `../docs/assets/historical-wrist-camera.png`、`historical-side-camera.png` | 从Spark既有`outputs/captured_images`只读复制的历史调试快照；不是本次重新开相机拍摄 |
| `../docs/assets/so101-muyu-scene.png` | 保留初始构图、按本机实拍结构修正后的木鱼任务生成式场景；不是本次真机实拍或成功率证据 |
| `../docs/assets/reference-photos/` | 七张本机实拍结构参考；用于校正生成式插图，不作为任务成功率证据 |
| `../docs/assets/legacy-vector/` | 风格统一前的代码绘制SVG草稿；已归档，不再与当前同名PNG视为一组 |

## 事实等级

- **直接文件证据**：V3配置、CSV首尾/统计、坏postprocessor仍存在Mac旧副本、旧打包覆盖语句、训练结束/最终导出日志；Spark当前修复配置与哈希、实际环境/补丁、历史60秒日志；V3全部帧future4与全局统计数值检查。
- **历史现场观察**：修复后的检查结果，以及“基本能完成”和掉棒等现象。不是新进行的受控实验。
- **合理推测/改进方向**：抓取停留短、恢复数据不足、视角变化影响；未做消融实验，不写为唯一根因。
- **未确认**：HF是否已发布Spark端postprocessor修复，当前Spark全部权重文件的重新计算哈希，独立场景成功率；文档明确保留未知。小型配置/统计文件哈希已核验，不等于完整权重重新校验。

## 官方参考（查阅于2026-09-12）

- [SO-101](https://huggingface.co/docs/lerobot/so101)
- [PI0](https://huggingface.co/docs/lerobot/pi0)
- [RTC](https://huggingface.co/docs/lerobot/rtc)
- [数据工具](https://huggingface.co/docs/lerobot/using_dataset_tools)
- [0.6.1 aggregate API](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/aggregate.py)
- [0.6.1 rollout配置](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/rollout/configs.py)
- [0.6.1 SO follower实现](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/so_follower/so_follower.py)
- [0.6.1 MotorBus标定与单位转换](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/motors/motors_bus.py)
- [0.6.1 Feetech homing offset实现](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/motors/feetech/feetech.py)

官方在线文档随版本演进。本教程锁定本实验0.6.1语境，新增脚本仍需在目标环境进行集成验收；不声称历史环境中的所有临时补丁均已恢复。

## 安全与隐私

未包含HF/W&B凭据、SSH私钥、原始示范视频或标定文件。展示的截图已人工查看，未见明文token。曾在对话里明文发送的凭据应轮换。不要把云镜像中的登录缓存、shell历史或`.netrc`一并分享。
