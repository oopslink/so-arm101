---
layout: article
title: 从照片到可抓握的木鱼仿真场景
subtitle: 零基础理解 MuJoCo 建模，并在 Mac 上运行 SO-101
section_label: 仿真建模
description: 从实物尺寸、局部坐标和 MJCF 几何体开始，搭建虎头木鱼、机械臂、双摄像头与物理接触抓握，并验证模型的边界。
---

# 从照片到可抓握的木鱼仿真场景：原理与 Mac 实操

本文带你理解并运行本仓库当前的 SO-101 木鱼仿真。无需先学会机器人学或三维建模，但需要能够打开终端、复制命令和编辑文本文件。

最终可以看到两种画面：一张按实物尺寸搭建的台面，以及机械臂通过夹爪接触夹住木槌、提起、敲击、抬起、张开释放的过程。

**当前演示开始时，木杆已被初始化在两指之间；机械臂会实际闭合夹持，但还不会从底板上接近并抓起木槌。** 演示由脚本控制，尚不是强化学习训练出来的动作。理解这个边界，才能正确判断模型做到了什么。

本文对应 2026-09-13 的本地实现。主入口是 `wooden_fish.play`，尺寸来自用户测量，部分形状、相机和接触参数仍为估算。

## 阅读路线

- 想先看结果：直接读[第 3 节](#3-在-mac-上从零运行)。
- 想知道场景怎么造出来：读第 4–9 节。
- 想自己修改：读第 10 节，再按第 11 节验证。
- 出现报错：查第 12 节。

## 1. 我们到底在建什么

把这个任务拆开，可以得到五层内容：

| 层次 | 要回答的问题 | 本项目的例子 |
|---|---|---|
| 外形 | 画面里是什么样子？ | 橙色虎头、白色槌头、绿色网格垫 |
| 空间 | 物体有多大，放在哪里？ | 木鱼 6×6×5.5 cm，在机械臂右侧 |
| 物理 | 会不会碰撞、滑动、掉落？ | 木槌有质量，夹爪松开后下落 |
| 控制 | 怎样让关节运动？ | 向电机模型发送目标角度 |
| 任务 | 怎样算完成？ | 提起、保持、敲击后仍夹持，张开后掉落 |

只有第一层，是一个三维展示。加上物理和控制，才有可交互的机器人仿真。再定义观测、动作、奖励和复位，才形成适合强化学习的环境。

本项目使用 **MuJoCo** 负责物理计算和渲染，使用 **Python** 组织模型、控制流程和验证。MuJoCo 的模型文件格式叫 **MJCF**，它是 XML 文本；你不需要先掌握 Blender，便能用基本几何体搭起一个场景。[MuJoCo 建模说明](https://mujoco.readthedocs.io/en/stable/modeling.html)

![当前台面几何预览](assets/simulation-modeling/tabletop.png)

*图 1：尺寸配置生成的静态台面。虎头是几何近似，底板上的木槌在这个预览中固定不动。画面左右取决于观察角度，机器人右侧需要用后面的世界坐标判断。*

## 2. 先认识项目文件：哪些是输入，哪些是输出

从仓库根目录看，核心结构如下：

```text
simulation/
├── assets/
│   ├── so101/                       官方机械臂模型、网格和许可证
│   └── tabletop/reference/          用户提供的实物照片
├── configs/
│   ├── tabletop-reference.json      尺寸、摆放和相机参数
│   └── grasp-physics.json           质量、摩擦和夹爪参数
├── wooden_fish/
│   ├── tabletop.py                  生成台面
│   ├── scene.py                     将台面与 SO-101 组合
│   ├── grasp.py                     将木槌改为自由物体并验证抓握
│   ├── view.py                      查看静态台面
│   ├── play.py                      打开实时动作演示
│   ├── run.py                       多回合验证与录像
│   ├── env.py                       历史固定持槌 Gymnasium 环境
│   └── train.py                     历史固定持槌 PPO 训练入口
├── tests/                           行为验证
└── runs/                            本机生成的图片、视频、指标和模型
```

`configs/` 是你通常应编辑的输入。`runs/` 是可重新生成的输出，已被 Git 忽略。不要把修改 `runs/tabletop-reference/scene.xml` 当作长期修改方式：下次生成时，它会被覆盖。

还有一个很容易误解的区别：

| 命令入口 | 实际做的事 | 有无真实接触抓握 |
|---|---|---|
| `wooden_fish.tabletop` | 生成静态台面的 XML 和三张图片 | 无，所有构件固定 |
| `wooden_fish.view` | 打开上述 XML，供旋转、缩放查看 | 无 |
| `wooden_fish.play` | 默认打开接触抓握演示窗口 | 有，物体是自由刚体 |
| `wooden_fish.grasp` | 验证接触抓握，可生成指标和录像 | 有 |
| `wooden_fish.run`，不传模型 | 默认多回合运行接触抓握脚本 | 有 |
| `play/run --fixed-grip` | 历史固定持槌脚本 | 无，杆固定在夹爪上 |
| `train` / `run --model ...` | 历史 RL 环境训练或按模型配置评估 | 仍是固定持槌基线 |

**台面 XML 本身不会“自动长出”机械臂。** 实时演示会重新读取配置，在内存中生成组合模型。两个程序共享建模代码，而不是始终打开同一个输出 XML。

## 3. 在 Mac 上从零运行

### 3.1 进入正确目录

打开 macOS 的“终端”，输入：

```bash
cd /Users/oopslink/works/codes/oopslink/so-arm101
pwd
```

第二条命令显示当前目录。应看到上面的完整路径。后文除特别说明外，命令都在这个目录运行；如果你把仓库放在其他地方，只需修改这一步。

终端前面的 `(base)` 是 Conda 环境提示。本文直接使用 `.venv-sim/bin/...`，因此会调用项目虚拟环境中的解释器，不依赖 `(base)` 中安装了什么。

### 3.2 检查或安装依赖

本机已经安装好环境时，先检查：

```bash
.venv-sim/bin/python --version
.venv-sim/bin/python -c 'import mujoco; print(mujoco.__version__)'
```

本文验证环境为 Python 3.12，MuJoCo 3.13.0。若命令能正常输出版本，直接跳到下一节，**不用重建环境**。

新机器可以通过 `uv` 准备 Python 3.12。`uv` 是 Python 环境与包管理工具。如果输入 `uv --version` 提示找不到命令，先按照 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)安装：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

这条命令下载并执行 uv 官方安装脚本。安装完成后重新打开终端，再执行第 3.1 节的 `cd`，然后运行：

```bash
uv --version
uv python install 3.12
```

第二条下载 Python 3.12，不会要求你替换 macOS 自带的 Python。[uv 的 Python 安装说明](https://docs.astral.sh/uv/guides/install-python/)

接着创建项目环境并安装依赖：

```bash
uv venv --python 3.12 .venv-sim
uv pip install --python .venv-sim/bin/python -e './simulation[test]'
```

第一条创建独立环境，第二条安装项目和测试依赖。`-e` 表示“可编辑安装”：修改本仓库 Python 文件后，下次运行就会使用新代码，无需重复安装。`[test]` 会安装测试所需的软件包。

如果你已经安装 Python 3.12，但没有 `uv`，可用 Python 自带的方法完成同样的工作：

```bash
python3.12 -m venv .venv-sim
.venv-sim/bin/python -m pip install -e './simulation[test]'
```

若 `python3.12` 也不存在，需要先安装该版本的 Python；安装解释器不是 MuJoCo 命令的一部分。不要在一个已经正常工作的 `.venv-sim` 上重复创建环境。

直接依赖在 [pyproject.toml](../simulation/pyproject.toml) 中固定版本。本机的完整包版本还记录在 `requirements-macos-arm64.lock.txt`；它面向本次 Mac 环境，不应直接当作 Spark 的 CUDA 环境锁。

### 3.3 生成静态场景

```bash
.venv-sim/bin/python -m wooden_fish.tabletop
```

这条命令依次完成：读取配置 → 创建几何模型 → 编译为 MuJoCo 模型 → 渲染图片 → 写出文件。

输出目录是：

```text
simulation/runs/tabletop-reference/
├── scene.xml
├── config.json
├── perspective.png
├── top.png
└── front.png
```

可以直接打开图片：

```bash
open simulation/runs/tabletop-reference/perspective.png
```

输出的 `config.json` 是生成当时的参数副本，方便对照图片；程序的默认输入仍是 `simulation/configs/tabletop-reference.json`。

### 3.4 打开可交互的台面窗口

```bash
.venv-sim/bin/mjpython -m wooden_fish.view
```

`mjpython` 是 MuJoCo 为 macOS 图形线程提供的启动器。我们写的 `view` 使用 `launch_passive`，因此 Mac 上用这个入口；官方独立查看器 `python -m mujoco.viewer` 属于另一种启动路径，不要混成 `mjpython -m mujoco.viewer`。[MuJoCo Python 查看器说明](https://mujoco.readthedocs.io/en/stable/python.html)

在窗口中可尝试左键拖动旋转、右键拖动平移、滚轮缩放。触控板的鼠标模拟可能与外接鼠标不同。查看器是“被动”的意思是 Python 负责更新数据和同步画面，不是物体一定静止；在 `view` 里，我们确实只查看固定摆放。

先检查：木鱼和白头是否位于底板上、木杆是否朝外伸出。然后关闭窗口。

### 3.5 运行接触抓握演示

```bash
.venv-sim/bin/mjpython -m wooden_fish.play
```

默认运行一回合，半速显示，结束后关闭。想按正常时间播放：

```bash
.venv-sim/bin/mjpython -m wooden_fish.play --speed 1
```

你会看到 `close → lift → hold → tap → retract → release`。最后松开并掉落是验证步骤：它用于证明木槌没有被软件固定在夹爪上。它不等于已经实现“稳稳放回底板”。

需要保存录像和指标时：

```bash
.venv-sim/bin/python -m wooden_fish.grasp \
  --video simulation/runs/contact-grasp.mp4 \
  --output simulation/runs/contact-grasp.json

open simulation/runs/contact-grasp.mp4
```

`--video` 指定视频文件，`--output` 指定指标文件。这条无窗口命令使用普通 `python`，无需 `mjpython`。视频会重写指定文件；保留不同实验时，应使用不同文件名。

## 4. 从照片变成模型：先分清实测与估算

照片主要用于确认轮廓、颜色、部件关系和方向。照片有透视，物体还高出网格平面，所以不能把像素距离直接当成准确尺寸。我们先做几何近似，再用用户提供的尺寸覆盖估算。

当前参数统一以米存储：

| 构件 | 用户数据 | 配置值 | 来源与解释 |
|---|---|---|---|
| 底板 | 13×7×0.8 cm | `0.13, 0.07, 0.008` | 实测 |
| 木鱼 | 6×6×5.5 cm | `0.06, 0.06, 0.055` | 实测包围尺寸 |
| 木杆 | 0.5×10 cm | 直径 `0.005`，长度 `0.10` | 0.5 按直径解释；10 是杆长 |
| 白头横向半径 | 0.6 cm | `0.006` | 用户提供 |
| 白头长轴 | 未给出 | `0.016` | 按照片形状估算 |
| 绿色垫板 | 未实测 | `0.45×0.30` | 近似尺寸 |
| 垫片、质量、摩擦 | 未实测 | 另见物理配置 | 工程初值 |

厘米转米就是除以 100。例如 `5.5 cm = 0.055 m`。本项目还使用千克、秒、牛顿及牛顿米，使质量、运动和受力处于同一套单位体系。

尺寸正确并不意味着形状精确。这里“木鱼宽 6 cm”是要求组合外形接近该尺度，不代表我们重建了真实内部空腔或每处曲面。

## 5. 坐标：怎样避免木鱼又放反

### 5.1 为什么必须定义“谁的右侧”

旋转查看窗口会改变画面左右，却不改变物体的位置。因此，约定如下：

- 世界 `+x`：从机械臂基座朝向台面。
- 世界 `+z`：向上。
- 站在机械臂后方、沿 `+x` 看向台面时，右侧是世界 `-y`。

当前木鱼在 `-y` 一侧，静置槌头在 `+y` 一侧。这个定义与用户最后确认的“木鱼在机械臂右侧”一致。

![当前机械臂与台面布局](assets/simulation-modeling/robot-layout.png)

*图 2：当前场景的斜俯视图。此图是接触演示初态，杆已位于两指之间；应以坐标判断左右，而不是根据截图边缘判断。*

### 5.2 台面还有自己的局部坐标

为了方便编辑，`tabletop.py` 先把台面放在自己的原点附近：底板平面中心为局部 x/y 原点，z=0 是绿色垫板顶面，局部 x 沿底板长边。

现在配置是：

```json
"fish": {
  "centre_xy": [0.03, 0]
},
"mallet": {
  "head_xy": [-0.035, -0.005],
  "handle_direction_y": -1
}
```

这只是摘录，不是完整配置，不能用它覆盖原文件。含义是木鱼在底板局部 `+x` 侧，静置槌头在另一侧，木柄沿局部 `-y` 伸出。

### 5.3 把整张台面放进机械臂世界

`scene.py` 再对整个台面应用：

```json
"robot_layout": {
  "plate_position": [0.28, 0, 0],
  "plate_yaw_degrees": -90
}
```

绕竖直轴转 -90° 后，局部 `+x` 变成世界 `-y`，局部 `-y` 变成世界 `-x`。所以：木鱼在机械臂右侧，木柄朝机械臂伸出。

对于这个特定角度，不必先学矩阵，也能直接计算：

```text
世界 x = 0.28 + 局部 y
世界 y =      - 局部 x
世界 z =        局部 z
```

木鱼底部中心的局部位置是 `(0.03, 0, 0.008)`，因此世界位置为 `(0.28, -0.03, 0.008)`。静置槌头中心 `(−0.035, −0.005, 0.014)` 则对应 `(0.275, 0.035, 0.014)`。

`plate_position` 的 0.28 m 仍是可达性验证用估算，并非已测出的基座到板中心距离。这里的 z 原点取垫板顶面，因此底板中心的实际高度还要加半厚度。

更一般的关系是 `世界位置 = 父物体平移 + 父物体旋转 × 局部位置`。MJCF 中 `body` 的位置相对父 body，几何体、相机和标记点的位置相对所在 body。把部件放在正确的父节点下面，就能让它随整体运动。[坐标与运动学树](https://mujoco.readthedocs.io/en/stable/modeling.html#coordinate-frames)

## 6. 基本几何体怎样组成木鱼和木槌

### 6.1 先读懂一小段 MJCF

下面是一个能够表示底板的最小例子，只展示底板，不是完整场景：

```xml
<mujoco model="plate_example">
  <compiler angle="radian"/>
  <worldbody>
    <body name="support_plate">
      <geom name="plate" type="box"
            pos="0 0 0.004"
            size="0.065 0.035 0.004"
            rgba="0.70 0.72 0.73 1"/>
    </body>
  </worldbody>
</mujoco>
```

逐项理解：

| 名称 | 本例中的作用 |
|---|---|
| `worldbody` | 整个世界的根节点 |
| `body` | 放置一组物体部件的局部坐标框架 |
| `geom` | 几何形状，可用于显示和碰撞 |
| `type="box"` | 形状为长方体 |
| `pos` | 形状中心的局部位置 |
| `size` | 与形状类型对应的尺寸参数 |
| `rgba` | 红、绿、蓝和透明度，数值在 0–1 之间 |

**长方体 `size` 写的是半尺寸。** 13×7×0.8 cm 的底板，必须写成 `0.065 0.035 0.004`。中心放在 z=0.004，底面才正好落在 z=0；若中心也放 z=0，就会有一半穿进垫板。

不同形状不能机械地使用相同规则：本项目椭球填三个半轴，球填半径，胶囊用半径和两个端点描述。具体属性以 [MJCF 几何体参考](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom) 为准。

### 6.2 虎头木鱼

`build_tabletop()` 没有读取虎头扫描网格，而是组合了若干基本形状：

- 椭球构成底部、上壳和两只耳朵。
- 较小的几何体构成眼睛、鼻子、口鼻区域和额头图案。
- 深色扁椭球提示底部开缝。
- 各部分尺寸按 `width/depth/height` 的比例生成。

例如上壳使用半轴 `(宽×0.5, 深×0.47, 高×0.355)`，中心位于高度 `高×0.565`。因此修改木鱼整体尺寸后，主要轮廓随之缩放。

开缝目前只是视觉近似，没有中空壳体和声学计算。现有仿真可以判断碰到木鱼，不能预测真实发声频谱或响度。

### 6.3 白头木柄槌

槌头用椭球，木杆用两端圆滑的胶囊体。台面预览中杆沿局部 y 方向；持槌模型中杆沿木槌自身局部 z 方向。局部轴不同，只是为了装配方便，不代表物体尺寸变了。

胶囊的两个端点是端部球心，因此：

```text
胶囊总长 = 两端球心距离 + 2 × 半径
```

直径 0.005 m、总长 0.10 m 的杆，两球心距离应为 0.095 m。代码会减去直径，避免把端部重复算入杆长。

当前白头横向半径为 0.006 m，长半轴为 0.008 m。杆插入白头多深还没有实测；预览杆与持槌杆使用相同外形尺寸，但连接处保留了装配近似。

### 6.4 外观几何与碰撞几何

眼睛、粉色耳部等装饰不应成为能卡住木槌的凸起。生成器为这些部分设置 `contype="0" conaffinity="0"`，让其不参与自动碰撞；壳体和主要部件保留碰撞。

渲染时隐藏第 3 组碰撞网格，只改变画面可见性，**不等于关闭物理碰撞**。注意，“关闭碰撞”和“忽略质量”也不是同一件事；自动质量推断还取决于模型的惯性设置。本项目机械臂使用上游惯性，木槌显式指定质量。

## 7. 加入 SO-101：复用官方模型

官方模型位于 [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)。仓库名包含 100，但也提供 SO-101 的 MJCF、URDF 和网格。

本项目将所用文件固定在提交 `eecbe3e0a9ebb23e25ad7b2759b03884c6660903`，保存于 `simulation/assets/so101/`，同时保留 `LICENSE`、`SOURCE.json` 和上游说明。固定版本能避免上游修改后，本地关节或几何悄悄变化。

上游提供新旧两种标定约定；当前使用 `so101_new_calib.xml`。官方也指出，LeRobot 夹爪的 0–100 开合表示不能直接当成这个仿真模型的关节角。[固定版本的官方说明](https://github.com/TheRobotStudio/SO-ARM100/blob/eecbe3e0a9ebb23e25ad7b2759b03884c6660903/Simulation/SO101/README.md)

组合过程是：

```text
官方 SO-101 XML
    + build_tabletop(尺寸配置)
    + 台面旋转、平移
    + 木槌和敲击标记
    + 两个相机
    ↓
build_tiger_model()
    ↓
build_grasp_model()：解除固定持槌，启用自由刚体与接触垫
    ↓
MuJoCo 编译模型，创建运行状态
```

`scene.py` 使用 Python 的 `xml.etree.ElementTree` 读取、增加和修改 XML 节点。上游文件保留原样，修改发生在内存中的组合模型上。

## 8. 怎样让夹爪真正夹住木槌

### 8.1 固定连接为什么看起来“抓住了”

如果把木槌写成夹爪下面一个没有关节的子 body，它会一直跟着夹爪走。夹爪张开，木槌也不会掉落。这种方式可以验证轨迹，但不能检验抓握。

当前接触版将木槌移到 `worldbody` 下，并加上：

```xml
<freejoint name="mallet_free"/>
```

它有三个平移自由度和三个旋转自由度，会受重力、惯性和碰撞影响。旋转状态用四元数存储，所以这个自由关节对应 7 个位置变量、6 个速度变量；这是表示方法的区别，不是物体多了一个物理自由度。

### 8.2 抓握靠什么保持

夹爪控制目标从初始 `−0.08 rad` 改为闭合 `−0.1745 rad`。电机模型尝试靠近目标，夹指遇到木杆后产生接触反力，实际角度会停在两者平衡处，而不是强行穿过杆。

两侧接触的法向力提供夹紧作用，摩擦阻碍杆沿夹指滑动。需要同时检查“接触力存在”和“杆没有明显滑移”，不能只看夹爪角度。

本次为 5 mm 细杆加入接触垫，并用背板长方体替代两个夹指的凸包碰撞。原因是凹形视觉网格的凸包可能覆盖实际空隙，产生不符合夹缝形状的接触。这里仍是简化碰撞模型，并未宣称复制了实物垫片的变形规律。

![实际物理仿真中的闭合夹持](assets/simulation-modeling/contact-grasp.png)

*图 3：木槌由两侧接触垫夹持。不是焊接、吸附或把杆姿态强行绑定到机械臂。*

### 8.3 哪些参数仍需实测

当前 `grasp-physics.json` 的质量为杆 12 g、头 8 g，滑动摩擦系数 1.0，夹爪力矩限值 0.4 N·m，接触垫半尺寸为 `0.0015, 0.007, 0.014` m。

这些是工程初值，不是测量报告。接触参数 `solref/solimp` 调整数值求解的软约束响应；它们不是橡胶硬度的直接同义词。把摩擦或夹紧力任意调大，可能让仿真更容易成功，却削弱实物参考价值。

### 8.4 控制怎样推进时间

本项目物理步长为 0.002 秒，控制程序每次推进 10 个物理步。因此物理计算约 500 Hz，控制更新约 50 Hz。

程序修改 `data.ctrl`，即执行器目标，再调用 `mj_step()` 让动力学更新位置和速度。`mj_forward()` 根据当前状态更新几何位置等派生量，本身不代表时间向前走了一步。

**初始化可以设置物体姿态；验证过程不能每帧把杆“摆回夹爪”。** 当前流程只在初态放置木槌，之后让自由物体通过物理积分运动。IK 只是根据目标位置计算关节角，最终仍由执行器和接触完成动作。[Python 数据与仿真接口](https://mujoco.readthedocs.io/en/stable/python.html)

## 9. 加入两个摄像头

相机要定义三件事：放在哪里、朝向哪里、视角多宽。当前位置与方向叫外参，镜头投影相关参数叫内参；真实相机还可能有畸变。

当前配置：

| 相机 | 位置 | 朝向参考点 | 竖直视场角 |
|---|---|---|---|
| `side` | 世界坐标 `(0.22, −0.30, 0.20)` m | 世界 `(0.25, 0, 0.06)` m | 55° |
| `wrist` | 夹爪局部 `(0, −0.04, −0.035)` m | 局部 `(−0.0079, 0, −0.20)` m | 70° |

侧相机挂在世界节点下，机械臂运动不会改变它的位置。臂上相机挂在 `gripper` 节点下，因此自动跟随夹爪。代码将 `look_at` 换算成相机的局部轴，再写入 MJCF 的 `xyaxes`；`look_at` 是本项目配置项，不是直接原样写入 XML 的属性。

![固定侧摄像头画面](assets/simulation-modeling/side.png)

*图 4：固定侧摄像头，使用当前估计安装位置。*

![随臂摄像头画面](assets/simulation-modeling/wrist.png)

*图 5：臂上相机。图像转角、遮挡与实物是否一致，仍需要真实图像和标定来核对。*

分别运行：

```bash
.venv-sim/bin/mjpython -m wooden_fish.play --camera side
```

关闭后再运行：

```bash
.venv-sim/bin/mjpython -m wooden_fish.play --camera wrist
```

这两个相机目前用于查看和录像，**还没有作为 RL 网络的双图像输入**。估计的视场角、分辨率 640×480 及局部安装偏移，也不能直接当作真实相机标定结果。相机定义不自动增加实体支架或质量。

## 10. 亲手修改一次场景

### 10.1 改什么文件

修改 [tabletop-reference.json](../simulation/configs/tabletop-reference.json)。JSON 是参数表：键名用双引号，数值不用引号，最后一项后面不能多一个逗号。它不支持直接插入 `//` 注释；说明写在已有 `notes` 中。

练习：把木鱼向机械臂右侧再移 5 mm。

```json
"centre_xy": [0.035, 0]
```

原值是 `[0.03, 0]`。因为台面旋转了 -90°，增加局部 x 会减小世界 y，所以这是向机械臂右侧移动。这个练习会改变用户确认的标准摆放，观察后请恢复为 `[0.03, 0]`。

### 10.2 修改后怎样生效

查看静态台面需要重新生成：

```bash
.venv-sim/bin/python -m wooden_fish.tabletop
.venv-sim/bin/mjpython -m wooden_fish.view
```

已有窗口不会自动热加载配置，先关闭再打开。接触演示则在创建模型时直接读取源配置，重新运行 `play` 即可；重新生成静态图片不是它的必要前置步骤。

可以先用无窗口验证检查模型是否仍能工作：

```bash
.venv-sim/bin/python -m wooden_fish.grasp \
  --output simulation/runs/after-edit.json
```

几何移动后若失败，先看是不是超出机械臂可达范围、槌碰到耳部或夹持发生滑移，不要马上修改成功阈值。

### 10.3 用副本练习，保留标准配置

如果只想试外观参数，可复制配置：

```bash
cp simulation/configs/tabletop-reference.json simulation/runs/my-tabletop.json
```

编辑副本后执行：

```bash
.venv-sim/bin/python -m wooden_fish.tabletop \
  --config simulation/runs/my-tabletop.json \
  --output simulation/runs/my-tabletop

.venv-sim/bin/mjpython -m wooden_fish.view \
  --scene simulation/runs/my-tabletop/scene.xml
```

这个副本只供静态生成器使用。当前 `play/grasp` 没有 `--config` 入口，仍读取标准源配置；不能因为副本图片改变，就认为接触演示也使用了副本。

## 11. 怎样判断模型真的有效

画面正确只是第一关。当前接触演示用以下条件联合判断：

| 指标 | 验收条件 | 检查目的 |
|---|---|---|
| 物体结构 | 自由木槌、无固定约束 | 排除软件绑住木槌 |
| 提起高度 | 大于 2.5 cm | 确认能带动物体上升 |
| 保持滑移 | 相对夹爪小于 5 mm | 排除缓慢滑落 |
| 双侧法向力 | 保持时两侧都大于 0.01 N | 确认夹持接触仍存在 |
| 木鱼接触 | 槌头接触目标壳体并产生正冲量 | 确认实际触碰 |
| 敲击后夹持 | 两侧有力、相对漂移小于 5 mm | 排除敲完就脱手 |
| 张开释放 | 槌头下降超过 5 cm | 确认能解除夹持 |

`tap_contact` 当前只验证与指定壳体的有效接触，**还不是“恰好一次清晰发声”检测器**。它不模拟声音，也没有完成完整三次计数任务。

终端最后的 JSON 中，`passed: true` 表示以上条件同时通过。`head_lift_m` 和 `release_drop_m` 单位是米，`min_pad_normal_force_N` 是牛顿，`tap_impulse_Ns` 是牛顿秒。一次成功不能说明对任意摆放或参数都可靠。

运行自动测试：

```bash
.venv-sim/bin/python -m pytest simulation/tests -q
```

当前实现有 17 项测试，覆盖接触流程、始终张开导致掉落的对照实验、相机随动、尺寸、旧配置兼容以及历史环境行为。这里的“17”是本版本记录，后续增加测试后数量可能变化。

本次文档编写时另执行了接触验证，结果保存在本机 `simulation/runs/docs-guide-validation.json`。旧版本记录中的小数指标对应当时布局，调整右侧摆放后应以新运行的报告为准。

## 12. 常见问题与定位顺序

| 现象 | 先检查什么 | 当前处理方法 |
|---|---|---|
| `No such file or directory` | 是否进入项目目录、环境是否存在 | 用 `pwd` 检查，再检查 `.venv-sim/bin/python` |
| `No module named wooden_fish` | 是否完成可编辑安装 | 使用项目解释器安装 `-e './simulation[test]'` |
| `ParseXML: Error opening file` | 场景是否生成、路径是否正确 | 先运行 `tabletop`，再运行自动定位默认文件的 `view` |
| `mjpython -m mujoco.viewer` 出现重复加载或 `_Simulate` 异常 | 是否混用两种启动模式 | 使用本项目 `mjpython -m wooden_fish.view` |
| 台面能显示，但不会动 | 运行的是 `view` 还是 `play` | `view` 只看固定摆放；动作演示用 `play` |
| 录像还是旧圆柱或夹爪不闭合 | 是否打开了旧文件、传了旧模型或 `--fixed-grip` | 不传模型运行当前 `run`，或直接打开 `play` |
| 改了参数窗口没变化 | 是否只改了输出副本、窗口是否未重启 | 改源配置并重建/重启 |
| 木鱼左右颠倒 | 当前视角是否旋转、修改的是局部还是世界坐标 | 核对世界 `-y` 为机器人右侧 |
| 杆从夹爪滑落 | 两侧是否有力、杆位置和碰撞形状是否正确 | 查看指标，再核对尺寸与接触参数 |
| 物体弹飞或明显穿透 | 初始重叠、形状和控制目标是否合理 | 先排除几何重叠，再调整数值参数 |
| `passed` 为 false | 具体是哪一项指标失败 | 不凭视频印象，也不直接放宽阈值 |

普通关节运动中应检查 `ctrl` 目标和实际角度的差异；细杆接触问题优先检查碰撞几何。不要用每帧改物体位置的办法“修复”掉落，那会让物理验证失去意义。

## 13. 接下来还缺什么

当前已具备：按实测尺寸的台面、官方机械臂、两个仿真视角、自由木槌与接触夹持，以及可执行的抓握保持/敲击/释放验证。

下一阶段要从真实初态开始：木槌先放在底板上，机械臂接近杆、闭合、提起，然后敲击并放回。这需要新的接近轨迹、抓取姿态、复位逻辑和失败判定，不能只在当前动画前面加一个“抓取”标签。

再之后，才将这个接触任务封装成包含夹爪动作的 RL 环境，训练抓取策略。视觉输入还要处理相机标定和观测接口。当前 `train.py` 是固定持槌基线；不要把它生成的 checkpoint 当作真实抓取模型。

## 源码与进一步阅读

- [台面生成器](../simulation/wooden_fish/tabletop.py)：几何形状与部件比例。
- [组合模型](../simulation/wooden_fish/scene.py)：台面变换、官方机械臂和相机。
- [接触抓握实现](../simulation/wooden_fish/grasp.py)：自由物体、接触垫、执行器控制与判分。
- [当前物理参数与验证说明](../simulation/CONTACT_GRASP.md)。
- [MuJoCo 建模指南](https://mujoco.readthedocs.io/en/stable/modeling.html)：模型树、坐标和约束。
- [MuJoCo XML 参考](https://mujoco.readthedocs.io/en/stable/XMLreference.html)：查询具体属性。
- [MuJoCo Python 文档](https://mujoco.readthedocs.io/en/stable/python.html)：加载、运行和查看器。

本文图片来自当前场景的实际渲染，生成说明见 [图片来源](assets/simulation-modeling/README.md)。它们不是实机效果，也不是 RL 策略的效果图。
