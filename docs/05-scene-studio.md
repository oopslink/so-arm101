---
layout: article
title: Scene Studio：用 MuJoCo 管理 SO-101 实验场景
subtitle: 把机器人、物体、相机、任务与验收条件放进同一个版本化场景
section_label: 场景实验室
description: SO-101 Lab 的 MuJoCo 场景管理模块：统一登记资产、相机、动作语义、任务阶段与成功条件，并提供检查、导出和渲染命令。
---

# Scene Studio：用 MuJoCo 管理 SO-101 实验场景

数据采集、模型训练、仿真验证和真机执行都在描述同一个实验，但它们很容易各自保存一份机器人、相机和任务配置。配置稍有错位，就可能出现画面方向不同、动作顺序不一致或仿真通过而真机失败。

Scene Studio 把这些信息集中成一个带版本的场景清单，并使用 MuJoCo 编译和检查物理模型。它不是浏览器里的小游戏，也不把近似模型称为已经标定的数字孪生。

![MuJoCo 中的 SO-101 与木鱼布局](assets/simulation-modeling/robot-layout.png)

*当前 MuJoCo 场景复用了官方 SO-101 模型，并按实拍场景近似建立底板、虎头木鱼、敲棒和双相机。*

## 1. 一个场景需要管理什么

```text
Scene manifest
├── Robot       SO-101 模型、关节顺序、控制模式
├── Objects     木鱼、敲棒、底板与台面尺寸
├── Physics     质量、摩擦、夹爪力矩与仿真步长
├── Cameras     overview、side、wrist
├── Task        指令、已实现阶段、未实现阶段
├── Variation   物体位置和初始关节扰动
└── Success     抬升、漂移、接触、掉落与敲击条件
```

第一版场景 ID 是 `wooden-fish-contact-v1`。清单位于 `simulation/configs/scenes/`，几何和相机配置、接触物理配置仍分别保存在原有 JSON 文件中；清单通过路径引用它们，避免复制参数。

场景中的六路动作按固定顺序表示底座、肩部、肘部、腕部俯仰、腕部旋转和夹爪的位置目标。注册表会检查动作数量是否与 MuJoCo 模型的六个执行器一致，也会检查侧相机、腕部相机、木鱼和敲棒是否真的存在于编译结果中。

## 2. 当前能验证什么

`wooden-fish-contact-v1` 使用自由刚体敲棒和夹指接触力，能够验证：

- 夹爪闭合后是否仅靠摩擦夹住木杆；
- 提起、保持和敲击过程中敲棒是否滑移；
- 槌头是否与木鱼目标壳体发生接触；
- 松开夹爪后敲棒是否在重力作用下掉落；
- 两路相机、动作维度和命名约定是否完整。

它还不能验证完整任务。当前没有实现视觉寻找、从底板接近并抓取、连续敲三下、准确放回和回位。这些阶段明确写在清单的 `not_implemented_phases` 中，避免把局部验证误解成端到端成功。

![固定侧相机视角](assets/simulation-modeling/side.png)

*固定侧相机用于观察机械臂、木鱼和敲棒的整体关系。位置和视场角目前仍是工程估计值。*

![腕部相机视角](assets/simulation-modeling/wrist.png)

*腕部相机跟随夹爪运动。仿真相机尚未完成与真实相机的内参、外参和畸变标定。*

## 3. 安装并检查场景

从仓库根目录执行：

```bash
uv venv --python 3.12 .venv-sim
uv pip install --python .venv-sim/bin/python -e './simulation[test]'

# 列出场景
.venv-sim/bin/python -m wooden_fish.scene_manager list

# 校验清单、资产路径、动作契约并编译 MuJoCo 模型
.venv-sim/bin/python -m wooden_fish.scene_manager validate

# 查看任务阶段和能力边界
.venv-sim/bin/python -m wooden_fish.scene_manager show wooden-fish-contact-v1
```

正常校验结果会包含：

```json
{
  "valid": true,
  "compiled": true,
  "model": {"nq": 13, "nv": 12, "nu": 6, "ncam": 2}
}
```

`nu=6` 表示模型接受六路执行器目标；`ncam=2` 表示 MJCF 内含侧面和腕部两个命名相机。总览视角是查看器相机，因此不计入 `ncam`。

## 4. 导出与渲染

```bash
# 导出编译后的标准 MJCF，同时保存对应场景元数据
.venv-sim/bin/python -m wooden_fish.scene_manager export wooden-fish-contact-v1 \
  --output simulation/runs/scene-studio/wooden-fish-contact-v1.xml

# 渲染固定侧相机
.venv-sim/bin/python -m wooden_fish.scene_manager render wooden-fish-contact-v1 \
  --camera side --output simulation/runs/scene-studio/side.png

# 渲染腕部相机
.venv-sim/bin/python -m wooden_fish.scene_manager render wooden-fish-contact-v1 \
  --camera wrist --output simulation/runs/scene-studio/wrist.png
```

`simulation/runs/` 是可重新生成的输出目录，不应把导出的 XML 当作配置源直接修改。持久变更应写入场景清单、台面配置或物理配置，然后重新校验和导出。

实时查看与接触动作验证继续使用：

```bash
.venv-sim/bin/mjpython -m wooden_fish.play --camera side
.venv-sim/bin/python -m wooden_fish.grasp \
  --video simulation/runs/contact-grasp.mp4
```

## 5. 后续场景版本

场景版本不会靠覆盖旧 JSON 演进。下一阶段建议分别建立：

1. `wooden-fish-pick-v1`：从底板接近、夹紧并提起敲棒。
2. `wooden-fish-three-taps-v1`：三次有效接触、间隔和重复碰撞判定。
3. `wooden-fish-full-task-v1`：寻找、抓取、三次敲击、放回和回位。
4. `wooden-fish-calibrated-v1`：写入实测相机外参、机器人到台面变换以及辨识后的摩擦和质量。

只有完成实物标定和多轮真机对照后，才适合把某个版本称为数字孪生。此前它的正确定位是：**可复现的工程场景和动作诊断工具**。

更完整的几何、坐标和接触建模说明见[仿真建模指南](03-simulation-modeling-guide.html)。
