# 机器人工作站只读核验

核验日期：2026-09-12。连接主机 `spark-f0d8`，只读取磁盘文件、Python包版本和历史文件；没有导入/调用机器人连接逻辑，没有打开摄像头、操作扭矩、运行策略或重新采集。将两份历史日志和两张既有相机调试快照复制到本项目。

## 1. 实际软件版本与环境入口

| 包 | 读取到的版本 |
|---|---|
| LeRobot | 0.6.1 |
| torch | 2.11.0+cu130 |
| torchvision | 0.26.0+cu130 |
| transformers | 5.5.4 |
| av | 15.1.0 |
| torchcodec | 0.11.1+cu130 |
| huggingface-hub | 1.28.0 |

现用环境入口是`<ROBOT_LEROBOT_ROOT>/activate.sh`：它不仅激活`.venv`，还把本地FFmpeg、CUDA和ARM64共享库目录加入环境。只激活`.venv`不能等价复现全部动态库环境。这些是本机读到的版本，不是给所有平台的通用安装锁。

## 2. 数据与future4全量数值核验

原合并目录：`<ROBOT_DATA_ROOT>/qiaomuyu_search_dual_v1_80_fixed`。

重标注目录：`<ROBOT_DATA_ROOT>/qiaomuyu_search_dual_v1_80_follower_future4`。

两者`meta/info.json`均为80 episodes、50,596 frames、30 FPS。本次读取重标注目录的全部data Parquet，跨文件按episode聚合后按frame_index排序，检查每条frame从0连续，逐帧验证：

```text
action[t] == observation.state[min(t+4, episode_last_frame)]
episodes checked: 80
frames checked: 50596
maximum absolute difference: 0
recomputed global action mean vs meta/stats.json: maximum difference 0
recomputed global action std  vs meta/stats.json: maximum difference 0
```

这是对这份实际V3数据的正面证据，不等于旧重标注脚本对任意分片数据都无缺陷。本次未逐帧解码视频、未重新判断演示成功与否，也未核验全部episode级统计。

## 3. 部署模型的当前状态

目录：`<ROBOT_MODEL_ROOT>/qiaomuyu_search_dual_pi0_v3_follower_future4`。

- postprocessor的action特征为`ACTION/[6]`，引用本次训练的unnormalizer统计文件。
- preprocessor的tokenizer_name是该模型目录下`tokenizer`的Spark绝对路径。
- `model.safetensors`文件大小为8,892,502,944字节；本次没有读取全部权重计算哈希或加载权重。

| 文件 | SHA256 |
|---|---|
| policy_postprocessor.json | `1411789f6ed6d69f80edaabc01db7a89bba70167bebd79bfd915021c360b8869` |
| policy_postprocessor_step_0_unnormalizer_processor.safetensors | `00547641e63583166908baa2b9972d6be9e91e0babde2e6d52f461f1ad20c2b0` |
| policy_preprocessor.json | `430114249e27541553340d4041004f603de7e2a2993c0092cfe5175bae586506` |
| config.json | `7f87e9592799adeb28c7db0416a78e91dcc0095baf3715b0c02916196c9e6dff` |

`pi0_trace_run.py`、`pi0_dataset_replay.py`、`init_safe_speed.py`的SHA256与本项目副本相同。采集脚本在本项目改成环境变量配置，因此不再要求字节一致。

HF远端模型是否已经同步修复，本次未查询。Mac旧副本与Spark修复副本的差异仍需在下一次发布时处理；这次没有修改任何模型工件。

## 4. 找到的批编码补丁

已安装文件：`<ROBOT_LEROBOT_ROOT>/.venv/lib/python3.12/site-packages/lerobot/datasets/dataset_writer.py`。

在`_batch_save_episode_video()`读取episode索引前，当前版本包含：

```python
# Episode metadata is buffered behind an open ParquetWriter. Close it before
# loading the rows needed by batch encoding, then refresh the in-memory view.
self._meta._close_writer()
self._meta.episodes = load_episodes(self._root)

chunk_idx = self._meta.episodes[start_episode]["data/chunk_index"]
file_idx = self._meta.episodes[start_episode]["data/file_index"]
```

同文件已导入`load_episodes`，写回episode元数据后也刷新内存视图。这里记录的是部署文件中的实际修复，不是重新执行该修复，也不是承诺这几行能修复所有版本、所有分片布局。没有原始备份时不能仅凭这个片段重建完整历史diff。

## 5. 历史60秒运行日志复核

本项目保留[原始文本日志](spark/qiaomuyu_pi0_v3_fixed_postprocessor_hardware_trace_60s.log)与[稀疏动作JSONL](spark/qiaomuyu_pi0_v3_fixed_postprocessor_hardware_trace_60s.jsonl)。均来自2026-09-10的历史运行，**不是本次重新实测**。

按日志原始时钟：21:54:06开始构建；21:55:38策略加载完成；21:55:41开始控制；21:56:41停止推理并开始回位；21:56:45结束。加载约92秒，不能把它混进60秒控制窗口。

JSONL共64行；step1到1740的59行属于回位前策略阶段，首末样本相隔58.483秒；step1770及以后处于回位阶段，绘图已排除。所有策略阶段保存样本的扭矩寄存器均为1，requested与sent最大差为0。只能说**已记录样本**中未见两者因裁剪而不同，不代表完整频率每一帧都已经检查。

| 关节 | 已记录actual最小值 | 最大值 | 单位 |
|---|---:|---:|---|
| shoulder_pan | 0.703 | 19.165 | 度 |
| shoulder_lift | 1.495 | 101.187 | 度 |
| elbow_flex | -55.385 | -1.758 | 度 |
| wrist_flex | -66.945 | 4.176 | 度 |
| wrist_roll | 2.242 | 17.275 | 度 |
| gripper | 2.162 | 12.500 | 0–100量纲 |

`actual`是最近观测缓存；稀疏日志不用于精确测量跟踪延迟。日志显示注册别名`so100_follower`，而已安装配置中`so100_follower`与`so101_follower`都注册到同一配置类，不能仅凭这个名字认定连接了错误机械臂。日志最终写入1911次包括回位，不应全部计入策略频率。

这些数值支持“该次修复后运行中机械臂已经产生明显动作”，不证明敲击次数或抓取成功。任务结果仍需结合当时的现场观察。

## 6. 历史双相机快照

教程中的`historical-wrist-camera.png`和`historical-side-camera.png`来自Spark既有的`outputs/captured_images/opencv__dev_video0.png`与`opencv__dev_video2.png`。本次按画面内容将近距离夹爪视角标为腕部相机、全局桌面视角标为固定侧视相机，没有重新打开设备验证动态画面。

原文件名中的`/dev/video0`、`/dev/video2`仅说明截图生成当时的Linux设备节点。USB重插和启动顺序可能改变节点号，因此教程只用它们展示视角分工，不把旧节点号写成稳定端口配置。
