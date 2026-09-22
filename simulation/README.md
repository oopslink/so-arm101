# 真实接触抓握（当前默认演示）

## Scene Studio：场景注册与校验

`configs/scenes/` 把任务意图、机器人动作顺序、双相机、几何配置、物理配置和成功条件登记为带版本的场景。它是配置与验证入口，不会把当前工程场景包装成已经标定的数字孪生。

```bash
# 查看、检查和编译所有已登记场景
.venv-sim/bin/python -m wooden_fish.scene_manager list
.venv-sim/bin/python -m wooden_fish.scene_manager validate

# 查看一个场景的能力边界
.venv-sim/bin/python -m wooden_fish.scene_manager show wooden-fish-contact-v1

# 导出可单独检查的 MJCF 与对应场景元数据
.venv-sim/bin/python -m wooden_fish.scene_manager export wooden-fish-contact-v1 \
  --output simulation/runs/scene-studio/wooden-fish-contact-v1.xml

# 从总览、固定侧相机或腕部相机渲染初始化画面
.venv-sim/bin/python -m wooden_fish.scene_manager render wooden-fish-contact-v1 \
  --camera side --output simulation/runs/scene-studio/side.png
```

当前注册的 `wooden-fish-contact-v1` 只实现夹紧、提起、保持、单次敲击、抬起和释放；视觉寻找、从底板接近抓取、连续敲三下、准确放回与回位仍明确列在 `not_implemented_phases` 中。

```bash
# Mac 实时演示，一回合结束后自动关闭；默认半速
.venv-sim/bin/mjpython -m wooden_fish.play

# 单独验证并生成物理指标与录像
.venv-sim/bin/python -m wooden_fish.grasp --video simulation/runs/contact-grasp.mp4

# 多回合录像，当前默认也采用真实接触抓握
.venv-sim/bin/python -m wooden_fish.run --episodes 3 \
  --video simulation/runs/scripted.mp4 --output simulation/runs/contact-evaluation.json
```

夹爪必须闭合才能夹持，张开后木槌在重力下掉落。没有焊接、固定连接、吸附或外加保持力。物体只在初始复位时放到两指之间；尚未实现从底板上接近并抓起。初始开度为 -0.08 rad，闭合目标 -0.1745 rad，夹到杆后实际关节停在接触平衡位置。

详细参数与验证见 [接触抓握说明](CONTACT_GRASP.md)。需要查看旧的固定持槌演示时显式加 `--fixed-grip`。下文 PPO 配置和旧指标仍描述历史固定持槌基线，**不能当作真实抓取策略的训练或成功率**。

---

# SO-101 单次敲木鱼仿真

当前实时演示和无模型录像默认使用接触抓握：闭合 → 提起 → 保持 → 敲击 → 抬起 → 张开掉落。木槌是自由刚体。`train` 和加载旧模型的 `run --model` 仍属于历史固定持槌 RL 基线，尚未迁移为抓取训练。
这里的程序只运行 MuJoCo，不连接机械臂、相机或串口。

## 当前状态

- 已在本机 macOS ARM64 / Python 3.12 验证环境、脚本控制、录像、PPO 短训练。
- 木鱼使用虎头椭球组合，槌固定连接在夹爪上；没有抓取、声音合成、放回或三次敲击。旧 checkpoint 若未记录场景类型，将明确加载历史圆柱环境。
- 观测使用仿真真实状态，尚未接视觉感知，也未对齐真机标定和夹爪单位。
- Spark 安装与吞吐尚未实测。不要把本机依赖锁直接当作 Linux CUDA 锁。
- 短训练用于验证流程，不能据此宣称策略已学会任务。详见 [验证记录](VALIDATION.md)。

## 从仓库根目录安装

本项目使用源目录 editable 安装；请保留 `simulation/assets/`，不将当前包单独构建成 wheel 分发。

```bash
uv venv --python 3.12 .venv-sim
uv pip install --python .venv-sim/bin/python -e './simulation[test]'
```

复现本次 macOS 依赖组合：

```bash
uv pip install --python .venv-sim/bin/python -r simulation/requirements-macos-arm64.lock.txt
uv pip install --python .venv-sim/bin/python --no-deps -e simulation
```

当前 MuJoCo、Gymnasium、SB3 等直接依赖在 `pyproject.toml` 中固定版本；PyTorch 由平台决定，实际版本写入每次训练记录。本机使用 CPU，不需要 CUDA。

## 验证场景和判分

```bash
.venv-sim/bin/python -m pytest simulation/tests -q
.venv-sim/bin/python -m wooden_fish.run --episodes 20 \
  --output simulation/runs/scripted-evaluation.json
.venv-sim/bin/python -m wooden_fish.run --episodes 3 \
  --video simulation/runs/scripted.mp4 \
  --output simulation/runs/video-evaluation.json
```

默认是位置 IK 脚本控制器：每步计算目标关节角，仍经位置执行器和真实仿真步执行，不在 episode 中直接改写关节状态。它用于验证任务可解，**不是 RL 策略**。增加 `--fixed` 可以关闭初始扰动。视频按 50 FPS 保存，单次动作很短。

## 训练、恢复、评估

所有输出路径必须使用新目录，避免覆盖已有实验。

```bash
# 流程测试：不是收敛训练
.venv-sim/bin/python -m wooden_fish.train --steps 4096 --envs 2 \
  --output simulation/runs/smoke

# 正式实验的起始预算，是否足够须看独立成功率
.venv-sim/bin/python -m wooden_fish.train --steps 200000 --envs 4 \
  --output simulation/runs/ppo-seed0

# 继续学习；加载模型与优化器，但重新开始 episode，不是逐位确定性恢复
.venv-sim/bin/python -m wooden_fish.train --steps 50000 --envs 4 \
  --resume simulation/runs/ppo-seed0/final_model.zip \
  --output simulation/runs/ppo-seed0-continued

# 使用未参与训练过程评估的另一组种子
.venv-sim/bin/python -m wooden_fish.run \
  --model simulation/runs/ppo-seed0/final_model.zip \
  --episodes 20 --seed 20000 \
  --output simulation/runs/ppo-test.json

.venv-sim/bin/tensorboard --logdir simulation/runs --host 127.0.0.1 --port 6006
```

PPO 使用 CPU、64×64 MLP、每环境 256 步 rollout。请求步数会按 rollout 批次向上取整。`--steps` 在恢复时表示额外训练步数。首次用 4 个环境，比较吞吐后再增加。

输出包含：`config.json`、checkpoint、最终模型、按验证奖励选择的 `best_model.zip`、Monitor CSV、TensorBoard 和评估记录。**最佳奖励不一定等于最高成功率**，最终验收看 `is_success`、失败原因及视频。模型加载要求同目录保留 `config.json`。观测未使用运行时归一化，部署时不存在遗漏 VecNormalize 统计的问题。

`config.json` 记录任务参数、随机种子、依赖版本、Git 版本、未提交状态和源文件 SHA-256。复现实验还需保留对应源文件；哈希不能恢复未提交代码。

## 环境约定

| 项目 | 当前值 |
|---|---|
| 物理步长 | 0.002 秒 / 500 Hz |
| 策略控制 | 每 10 个物理步 / 50 Hz |
| 动作 | 5 个关节目标增量，[-1,1] 映射至 ±0.015 rad/步 |
| 夹爪 | 固定目标 0.4 rad，槌为刚性固定附件 |
| 观测 | 32 维：6 关节位置/速度、5 控制目标、槌头相对位置/速度、上一动作、敲击计数/分离时长/进度/接触状态 |
| 起始扰动 | 木鱼 x/y ±5 mm、初始关节 ±0.003 rad |
| 超时 | 250 个控制步，5 秒 |
| 有效区域 | 虎头冠部暂定目标周围 12 mm；旧圆柱为 35 mm |
| 有效下落速度 | 0.025–0.7 m/s |
| 成功 | 有效敲击后抬起，槌头底部离顶面 ≥25 mm，连续保持 40 ms |

这些数值是工程初值，未经过实物辨识。`reset()` 用 IK 在目标上方初始化并预运行 0.2 秒，属于容易的局部任务，不能代表寻物或位置泛化能力。

奖励：距离进展×10、首次有效敲击+3、抬起完成+10、额外敲击或其他碰撞−5、每步−0.002，以及动作变化惩罚。距离目标在敲击后切换为抬起目标。每个物理步检查接触，持续接触不重复计数。非槌头—有效顶面的其他接触结束任务；回弹后重新接触也会失败，属于严格单次敲击定义。

## 模型来源与范围

`assets/so101/` 来自 TheRobotStudio/SO-ARM100，固定到
`eecbe3e0a9ebb23e25ad7b2759b03884c6660903`。
保留原始 MJCF、网格、Apache-2.0 LICENSE、上游说明和 `SOURCE.json`；没有改动上游文件。

`wooden_fish/scene.py` 在内存中加入物体、照明和接触设置。对固定槌与相邻夹爪部件排除接触；官方模型本身未包含底座碰撞体。视频隐藏碰撞网格，只显示外观网格。

## Spark 部署

将本仓库（包括未提交的 `simulation/` 文件）同步到 Spark，在独立 `.venv-sim` 中安装。
第一版 PPO 使用 CPU，先验证 PyTorch ARM64、MuJoCo 导入和无渲染测试，不需要为训练安装 Isaac Sim 或更改 CUDA 驱动。

Linux 无窗口录像可以先尝试 `MUJOCO_GL=egl`；它需要可用的 EGL 驱动，导入 MuJoCo 前设置。若失败，先完成无渲染训练，再诊断 EGL，或将 checkpoint 和配置复制到 Mac 录像。当前没有在 Spark 验证此路径。

TensorBoard 只绑定远端 `127.0.0.1`，通过 SSH 的 `-L 6006:127.0.0.1:6006` 转发访问；长训练可在 `tmux` 中运行。

## 文件入口

- `wooden_fish/scene.py`：组合官方模型与场景。
- `wooden_fish/env.py`：任务配置、观测、奖励、复位和物理步。
- `wooden_fish/hit_detector.py`：敲击与分离状态机。
- `wooden_fish/run.py`：脚本验证、模型评估、录像。
- `wooden_fish/train.py`：PPO、日志、checkpoint 和恢复。
- `tests/test_environment.py`：Gym 接口、接触计数、失败路径、脚本动作回归。

## 根据实物照片制作的台面预览

新增 [照片与建模说明](assets/tabletop/README.md) 和 [可调尺寸配置](configs/tabletop-reference.json)。用户已确认第五张照片是实际初始摆放，以此作为组装参考：虎头木鱼位于银色底板一侧，白色槌头搁在另一侧，木柄伸出底板；未添加照片中没有的独立槌架。

```bash
.venv-sim/bin/python -m wooden_fish.tabletop
```

输出位于 `simulation/runs/tabletop-reference/`：`perspective.png`、`top.png`、`front.png`、`scene.xml` 和使用的 `config.json`。也可以通过 `--config` 指定自己的尺寸配置。

这是固定构件的外形/布局预览，已应用用户提供的底板、木鱼、杆和白头半径尺寸；白头长轴、构件偏移和曲面细节仍为估算。它已与默认敲击环境复用几何；单次敲击从持槌状态开始，尚未实现第五张初始摆放到抓取的过程。实际敲击区域和底板相对机械臂的位置仍待确认。

### Mac 交互查看台面

生成场景后，在仓库根目录运行：

```bash
.venv-sim/bin/mjpython -m wooden_fish.view
```

这个入口用绝对路径加载默认场景，并调用 `launch_passive`，设置适合台面的初始视角。它只用于查看固定摆放，不推进训练或机械臂控制。已在本机通过 `--seconds 3` 验证窗口创建和自动关闭。

不要使用 `mjpython -m mujoco.viewer`：本机该组合出现模块重复加载警告与 `_Simulate` 初始化异常。官方独立查看器的命令是普通 `python -m mujoco.viewer`；macOS 下的 `mjpython` 用于这里的 passive viewer 入口。


## Mac 实时敲击（虎头场景）

```bash
.venv-sim/bin/mjpython -m wooden_fish.play
```

窗口重复显示“持槌准备 → 敲一下 → 抬起”，每回合前后停留一秒供观察，默认半速播放。关闭窗口结束；`--episodes 1` 只演示一回合，`--speed 1` 实时播放。

`wooden_fish.view` 查看第五张的台面初始摆放；`wooden_fish.play` 查看机械臂已经持槌后的动作。两者共用台面生成器，后者将底板上那把槌移到夹爪，不保留重复道具。`wooden_fish.run` 默认也已改为虎头场景，录像仍可写入 `runs/scripted.mp4`。

底板相对机械臂的平移暂设为 `[0.28, 0, 0]` m，冠部敲击目标是可行性验证用暂定位置；并非实物外参。白头长轴、持槌姿态及质量仍为估算。头部竖直支撑半径按椭球当前朝向计算，成功需要接触目标壳体并抬起。

新的训练配置保存 `scene_profile` 和台面尺寸快照；加载历史没有 `scene_profile` 的配置时仍使用圆柱场景，防止悄悄改变旧模型的评估环境。

## 标准布局与双摄像头

用户照片 `assets/tabletop/reference/06-standard-layout.jpg` 确认了台面长边与机械臂接近方向垂直、木杆朝机械臂伸出，以及固定侧摄像头和臂上摄像头。组合场景将台面绕 z 轴旋转 -90°；平移仍为暂定的 `[0.28, 0, 0]` m。台面预览使用自身坐标系，所以它的朝向不随此世界坐标变换改变。

实时查看两个摄像头：

```bash
.venv-sim/bin/mjpython -m wooden_fish.play --camera side
.venv-sim/bin/mjpython -m wooden_fish.play --camera wrist
```

默认 `--camera overview` 保留可自由旋转的总览。侧摄像头固定于世界，臂上摄像头绑定官方模型的 `gripper` 部件。安装偏移、光轴和垂直视场角保存在 `configs/tabletop-reference.json` 的 `cameras` 中，全部为估算，尚未标定内参、外参、畸变或画面旋转方向。渲染分辨率目前为 640×480，也不是实机采集分辨率的声明。

录像同样支持 `wooden_fish.run --camera side` / `--camera wrist`。相机不增加碰撞实体或安装件质量；当前仅提供视角，PPO 的 32 维状态观测尚未改为双图像输入。
