# 接触抓握实现与验证

## 实现范围

`wooden_fish/grasp.py` 实现闭合、提起 4 cm、保持 2 秒、接触敲击、抬起、张开释放。

- 木槌从夹爪的子 body 移至 world，并添加 `freejoint`（6 个自由度）。
- 去掉木槌与夹指之间的接触排除。没有 weld/equality、吸附、外加保持力或运行时写入木槌姿态。
- 第六个关节执行器真实控制夹爪闭合与张开。接触反力让它停在约 -0.156 rad，而不是强制穿过木杆达到目标 -0.1745 rad。
- 上游两个凹形夹指网格在 MuJoCo 中使用凸包碰撞时，会在细杆附近产生不合理的凸包接触。仅这两个网格的碰撞由显式夹指背板和黑色接触垫长方体替代；视觉网格和其他机械臂碰撞保留。
- 两侧接触垫摩擦与接触力承担木槌重量。夹爪张开后木槌自由下落。
- `grasp_probe` 是无质量的 IK 计算点，更新它只用于求关节目标，不约束木槌、不施加力。

**开始时木杆被初始化在两指之间，然后闭合。没有实现从第五张底板初态开始的接近和抓取。** 木槌初始化后，轨迹中只写机器人执行器目标。

## 参数与现实差距

`configs/grasp-physics.json` 保存工程参数：总质量 20 g（杆 12 g、头 8 g）、滑动摩擦系数 1.0、夹爪力矩限值 0.4 N·m、接触垫厚 3 mm、宽 14 mm、长 28 mm。这些尚未经过实物测量或辨识。

杆直径、杆长和白头半径沿用用户尺寸；白头长轴仍为估算。该版本验证的是物理接触流程，不代表真实机械臂上必定能承受相同敲击。

## 运行

从仓库根目录：

```bash
.venv-sim/bin/mjpython -m wooden_fish.play
.venv-sim/bin/mjpython -m wooden_fish.play --camera wrist --speed 1
.venv-sim/bin/python -m wooden_fish.grasp --video simulation/runs/contact-grasp.mp4
.venv-sim/bin/python -m wooden_fish.run --episodes 3 --output simulation/runs/contact-evaluation.json
```

`play` 默认运行一个完整接触验证回合；末尾张开放手是验证步骤，不是意外掉槌。`run` 不传模型时默认同样使用接触脚本；历史固定持槌录像需要 `--fixed-grip`。`run --model` 仍按模型保存的旧配置执行。

当前接触控制器是脚本 + IK，尚未封装成六动作 RL 抓取任务。`train` 仍训练历史固定持槌环境，不能用旧模型的成功率代表真实抓握。

## 本机实测

2026-09-13，macOS ARM64，现有 `.venv-sim`。

- 17 项测试通过，包括 3 个种子的完整接触流程。
- 张开夹爪对照测试：木槌下降超过 5 cm，没有残留夹持力。
- 默认种子下：提起 3.967 cm，保持相对滑移 0.476 mm，双侧法向力分别至少 8.06 / 7.88 N。
- 有效壳体接触冲量约 0.0531 N·s；敲击后相对滑移 0.727 mm，仍有双侧夹持力。
- 张开放手后槌头下降 9.38 cm。
- `play --episodes 1 --speed 2` 已在本机完成窗口验证；录像与夹持近景已检查。

`runs/contact-grasp.json` 和 `runs/contact-evaluation.json` 为本机完整输出；关键结果归档在 `contact-grasp-validation.json`。

成功要求：自由物体且无 equality、提起超过 2.5 cm、保持漂移小于 5 mm、两侧持续有接触力、敲击后仍保持夹持、张开后下降超过 5 cm。这不是完整桌面抓取成功率，也没有执行实机动作。
