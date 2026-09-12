# nero_ros2_lerobot_0.4.4


本仓库基于 **LeRobot 0.4.4** 修改，用于适配 **Nero 7-DoF 机械臂 + ROS2 + Pika 遥操作设备**，实现机器人数据采集、数据集清洗、时序对齐、ACT 策略训练及模型部署。

主要流程：

```text
ROS2 遥操作
    ↓
LeRobot 数据采集
    ↓
无效 Episode 清洗
    ↓
多模态数据时序对齐 / 30 Hz 重采样
    ↓
数据集划分
    ↓
ACT 模型训练
    ↓
模型部署
```

---

## 1. 数据采集

使用自定义的 `nero_ros2` Robot 和 Teleoperator 接口进行数据采集。

```bash
lerobot-record \
  --robot.type=nero_ros2 \
  --robot.id=nero \
  --teleop.type=nero_ros2 \
  --teleop.id=pika \
  --dataset.repo_id=szy/nero_ge \
  --dataset.root=/home/szy/lerobot_data/nero_ge_100hz \
  --dataset.single_task="move_box" \
  --dataset.fps=100 \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=1 \
  --dataset.push_to_hub=false \
  --display_data=false \
  --resume=true
```

### 1.1 实机 F8 自动复位

实机采集时显式选择 `real` 复位模式，并设置七个关节的复位目标（单位 rad）：

```bash
lerobot-record \
  --robot.type=nero_ros2 \
  --robot.id=nero \
  --robot.reset_mode=real \
  --robot.reset_joint_positions='[0.0,-1.74,0.0,2.14,0.0,0.0,1.57]' \
  --robot.real_reset_timeout_s=30 \
  --teleop.type=nero_ros2 \
  --teleop.id=pika \
  ...
```

按 F8 后，程序会结束当前 episode，关闭实机控制门、暂停 Pika 遥操作，向
`/control/move_j` 发布一次复位目标并等待关节稳定；随后在控制门关闭期间重新建立
Pika/TCP 基准，确认恢复后的首个 IK 目标接近实机当前位置，最后才恢复控制并开始下一轮。
任何步骤失败时控制门保持关闭，需要排查后手动调用 `/control_enable` 恢复。

实机模式默认不会在采集程序启动时自动运动。如需第一轮开始前也自动复位，增加：

```text
--robot.reset_on_start=true
```

仿真继续使用默认的 `--robot.reset_mode=simulation`，F8 会调用 `/reset_episode`；完全
禁用 F8 的机器人/环境复位可使用 `--robot.reset_mode=disabled`。

主要参数：

* `dataset.repo_id`：LeRobot 数据集 ID
* `dataset.root`：本地数据存储目录
* `dataset.fps`：目标采集频率
* `dataset.num_episodes`：计划采集 Episode 数量
* `dataset.episode_time_s`：单轮最长采集时间
* `dataset.reset_time_s`：Episode 之间的复位时间
* `resume=true`：在已有数据集基础上继续采集

---

## 2. 删除无效 Episode

例如删除第 `25、26、27、61` 轮无效数据：

```bash
lerobot-edit-dataset \
  --repo_id szy/nero_ge \
  --root /home/szy/lerobot_data/nero_ge_100hz \
  --new_repo_id szy/nero_ge_clean \
  --operation.type delete_episodes \
  --operation.episode_indices "[25, 26, 27, 61]"
```

---

## 3. 数据时序对齐与重采样

原始数据可按较高频率采集，再离线对齐并重采样至 ACT 实际训练使用的频率，例如 **30 Hz**。

### 3.1 Episode 分文件存储

当不同 Episode 的数据分别存储在多个文件中时：

```bash
python /home/szy/lerobot-0.4.4-clean/src/lerobot/scripts/nero_ros2_resample_dataset.py \
  --source-root /home/szy/lerobot_data/nero_ge_100hz/szy/nero_ge_clean \
  --output-root /home/szy/lerobot_data/nero_ge_30hz_aligned \
  --output-fps 30 \
  --episode all \
  --max-action-age-ms 40
```

检查时序偏差：

```bash
python /home/szy/lerobot-0.4.4-clean/src/lerobot/scripts/nero_ros2_plot_timestamp_offsets.py \
  --dataset-root /home/szy/lerobot_data/nero_ge_30hz_aligned \
  --episode all
```

### 3.2 多轮数据连续存储

当多个 Episode 的原始数据连续存储于同一数据文件中时：

```bash
python /home/szy/lerobot-0.4.4-clean/src/lerobot/scripts/nero_ros2_resample_shared_episode_dataset.py \
  --source-root /home/szy/lerobot_data/nero_ge_100hz/szy/nero_ge_clean \
  --output-root /home/szy/lerobot_data/nero_ge_30hz_aligned \
  --output-fps 30 \
  --episode all \
  --max-depth-delta-ms 20 \
  --max-fisheye-delta-ms 20 \
  --max-state-delta-ms 50 \
  --max-action-age-ms 40
```

主要同步阈值：

```text
Depth camera    ≤ 20 ms
Fisheye camera  ≤ 20 ms
Robot state     ≤ 50 ms
Action age      ≤ 40 ms
```

检查对齐结果：

```bash
python /home/szy/lerobot-0.4.4-clean/src/lerobot/scripts/nero_ros2_plot_shared_episode_offsets.py \
  --dataset-root /home/szy/lerobot_data/nero_valid_30hz \
  --episode all
```

---

## 4. 数据集划分

将清洗并对齐后的数据集按照 `8:1:1` 划分为训练集、验证集和测试集：

```bash
lerobot-edit-dataset \
  --repo_id szy/nero_ge_clean \
  --root /home/szy/lerobot_data/nero_ge_30hz_aligned \
  --operation.type split \
  --operation.splits '{"train": 0.8, "val": 0.1, "test": 0.1}'
```

---

## 5. ACT 模型训练

训练入口：

```bash
python /home/szy/lerobot-0.4.4-clean/src/lerobot/scripts/nero_ros2_train_act.py \
  --config /home/szy/lerobot-0.4.4-clean/src/lerobot/scripts/nero_ge_act_train_config.json
```

训练配置位于：

```text
src/lerobot/scripts/nero_ge_act_train_config.json
```

可根据不同数据集、图像输入、训练步数和模型参数创建新的配置文件。

仅进行数据集和模型配置验证时：

```bash
python /home/szy/lerobot-0.4.4-clean/src/lerobot/scripts/nero_ros2_train_act.py \
  --config /home/szy/lerobot-0.4.4-clean/src/lerobot/scripts/nero_ge_act_train_config.json \
  --validate-only
```

正式训练时需要去掉：

```text
--validate-only
```

---

## 6. 训练曲线分析

```bash
python /home/szy/lerobot-0.4.4-clean/src/lerobot/robots/so_follower/plot_lerobot_training_log.py \
  /home/szy/lerobot-0.4.4-clean/outputs/train/nero_ge_clean_act_training.log \
  --output-dir /home/szy/lerobot-0.4.4-clean/outputs/train/nero_ge_clean_act_color_fisheye_30hz/curve_analysis
```

---

## 7. ACT 模型部署

### 7.1 MuJoCo 仿真环境

在仿真工程主目录运行：

```bash
python3 18mujoco_single.py \
  --ros-args \
  -p mjcf_path:=/home/szy/agx_arm_sim/agx_arm_description/mujoco/nero_gripper_d435_antislip.xml \
  -p keyframe_name:=home
```

### 7.2 调整机械臂关节

例如设置 `joint4`：

```bash
ros2 topic pub --once \
  /control/joint_states \
  sensor_msgs/msg/JointState \
  "{name: ['joint4'], position: [2.07]}"
```

### 7.3 调整夹爪

```bash
ros2 topic pub --once \
  /sensor/gripper/joint_state \
  sensor_msgs/msg/JointState \
  "{name: ['sensor_gripper_center_joint'], position: [0.0967]}"
```

### 7.4 部署 ACT Policy

```bash
python /home/szy/lerobot-0.4.4-clean/src/lerobot/scripts/nero_ros2_deploy_act.py \
  --checkpoint /home/szy/lerobot-0.4.4-clean/outputs/train/nero_act_color_fisheye_30hz/checkpoints/100000/pretrained_model \
  --device cuda \
  --fps 30 \
  --duration-s 30 \
  --n-action-steps 1 \
  --send-actions
```

主要参数：

```text
--checkpoint       模型 checkpoint
--device cuda      使用 GPU 推理
--fps 30           Policy 推理 / 控制频率
--duration-s 30    单次部署持续时间
--n-action-steps 1 每次推理执行的 action 数量
--send-actions     实际向机器人发送动作
```

调试时如不希望机器人真实运动，应注意是否启用 `--send-actions`。

---

## 8. 上传数据集至 Hugging Face

首先登录：

```bash
hf auth login
```

在 Hugging Face 中创建具有写权限的 Access Token，并输入 Token。

上传数据集：

```bash
hf upload nofrowns/act \
  /home/szy/lerobot_data/nero_30hz/episode1 \
  . \
  --repo-type dataset
```

其中：

```text
nofrowns/act
```

表示：

```text
用户名 / 数据集仓库 ID
```

Hugging Face CLI 可根据指定的 Repo ID 创建对应数据集仓库。

---

## 9. 推荐工作流程

```text
1. 启动 Nero ROS2 与 Pika 遥操作
        ↓
2. lerobot-record 采集原始数据
        ↓
3. 删除失败 / 无效 Episode
        ↓
4. 离线进行多模态时间戳对齐
        ↓
5. 重采样至 30 Hz
        ↓
6. 绘制 timestamp offset 检查同步质量
        ↓
7. 划分 Train / Val / Test
        ↓
8. 训练 ACT
        ↓
9. 分析训练曲线
        ↓
10. MuJoCo / 真实 Nero 部署验证
```

---

## Notes

本仓库针对 Nero ROS2 接口进行了定制修改，部分脚本、话题名称、数据格式及路径与官方 LeRobot 0.4.4 不同。

当前命令中的 `/home/szy/...` 为本机开发环境路径，在其他设备部署时需要根据实际安装位置修改。
