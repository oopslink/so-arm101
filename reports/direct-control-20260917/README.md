# 2026-09-17 直接控制实验审计工件

报告正文：[`docs/04-direct-control-experiment-20260917.md`](../../docs/04-direct-control-experiment-20260917.md)。

- `summary.json`：69份日志、60段下发与80条训练示教的统计；57段无控制异常不等于57次任务成功。
- `runs.json` / `runs.csv`：逐段时序、跟踪误差、停止采样及异常。tap的target字段是敲击端点，实际命令包含往返，以command序列为准。
- `log-manifest.json`：本次读取的原始控制日志文件名、大小、SHA256。
- `training-metadata.json`：训练数据metadata、Parquet相对路径和SHA256。原数据与future4的state相同。
- `spark-direct-history-final-20260917.jsonl`：历史目标/结束状态，不能代替早期不存在的逐帧日志。
- `controller-*-snapshot.py.txt`：历史控制代码快照，仅供审计，不建议直接执行；保留当时路径和配置，不含凭据。
- `training-export-manifest.json`：本地数值导出的哈希与大小；完整训练数值不公开随Git发布。

完整原始数据本地目录：`/Users/oopslink/Documents/robot-logs/20260917-203612/`。

复现需要numpy、matplotlib和本地原始日志/训练NPZ；训练导出另外需要pandas、pyarrow。两个分析/导出脚本均不连接机器人串口。训练数据的timestamp是标称采样时间轴，不能据此验证历史硬件循环的真实30Hz频率。

图表的数值差分在每个episode/运动段内独立进行，统一25Hz和200ms窗口，排除夹爪。跨段空档单独计算，不插值成运动，也不与训练episode长度当作受控耗时对比。
