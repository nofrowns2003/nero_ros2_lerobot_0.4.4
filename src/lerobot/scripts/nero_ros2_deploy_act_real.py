#!/usr/bin/env python3

"""Deploy a Nero ACT policy on the real arm with temporal ensembling.

Unlike the simulation-oriented deployment entry points, this script treats the
recording reset pose as an explicit real-hardware precondition.  Moving to that
pose is opt-in, and every hardware command is guarded by topic, freshness, and
timing checks.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from sensor_msgs.msg import JointState

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.robots.nero_ros2.config_nero_ros2 import NeroRos2Config
from lerobot.robots.nero_ros2.nero_ros2_deployment import NeroRos2Deployment
from lerobot.scripts.nero_ros2_deploy_act import (
    apply_safety_limits,
    load_training_bounds,
    make_dataset_features,
    observation_state,
    resolve_device,
    validate_policy_features,
    wait_for_control_subscribers,
)
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action


DEFAULT_CHECKPOINT = Path(
    "/home/szy/lerobot-0.4.4-clean/outputs/train/"
    "nero_z_clean_act_color_fisheye_30hz/checkpoints/120000/pretrained_model"
)
DEFAULT_START_JOINT_POSITIONS = (0.0, -1.50, 0.0, 2.14, 0.0, 0.0, 1.57)
# The recording dataset starts with the gripper nearly fully open.
DEFAULT_START_GRIPPER_POSITION = 0.0964
COMMAND_DISCOVERY_SETTLE_S = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nero ACT 实机时间集成部署：state + color + fisheye，深度保持运行但不参与推理。"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=1,
        help="兼容原部署接口；时间集成需要逐帧推理，因此有效值固定为 1。",
    )
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=0.01)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只读取、推理和打印，不发送任何控制。")
    mode.add_argument("--send-actions", action="store_true", help="向实机发送经过安全限制的位置动作。")

    parser.add_argument(
        "--start-joint-positions",
        type=float,
        nargs=7,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        default=DEFAULT_START_JOINT_POSITIONS,
        help="采集时使用的 7 轴复位目标。",
    )
    parser.add_argument(
        "--start-gripper-position",
        type=float,
        default=DEFAULT_START_GRIPPER_POSITION,
        help="部署起始夹爪开度；默认取训练 episode 首帧附近的开爪值。",
    )
    parser.add_argument(
        "--move-to-start",
        action="store_true",
        help="正式推理前通过 /control/move_j 和夹爪控制话题移动到起始姿态；必须同时使用 --send-actions。",
    )
    parser.add_argument("--start-tolerance-rad", type=float, default=0.03)
    parser.add_argument("--start-gripper-tolerance", type=float, default=0.005)
    parser.add_argument("--start-stable-time-s", type=float, default=0.5)
    parser.add_argument("--move-timeout-s", type=float, default=30.0)
    parser.add_argument("--move-republish-s", type=float, default=2.0)
    parser.add_argument(
        "--allow-start-mismatch",
        "--allow-ood-start",
        dest="allow_start_mismatch",
        action="store_true",
        help="允许当前反馈偏离采集起始姿态；--allow-ood-start 是兼容旧接口的别名。",
    )

    parser.add_argument("--max-arm-step-rad", type=float, default=0.05)
    parser.add_argument("--max-gripper-step", type=float, default=0.005)
    parser.add_argument("--observation-timeout-s", type=float, default=0.5)
    parser.add_argument("--max-control-lag-s", type=float, default=0.25)
    parser.add_argument("--subscriber-timeout-s", type=float, default=5.0)
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--node-name", default="lerobot_nero_ros2_act_real_deploy")
    parser.add_argument(
        "--hold-on-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="发送过实机命令后，退出前发布一次当前反馈位置作为保持目标（默认启用）。",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "--fps": args.fps,
        "--duration-s": args.duration_s,
        "--start-tolerance-rad": args.start_tolerance_rad,
        "--start-gripper-tolerance": args.start_gripper_tolerance,
        "--start-stable-time-s": args.start_stable_time_s,
        "--move-timeout-s": args.move_timeout_s,
        "--move-republish-s": args.move_republish_s,
        "--max-arm-step-rad": args.max_arm_step_rad,
        "--max-gripper-step": args.max_gripper_step,
        "--observation-timeout-s": args.observation_timeout_s,
        "--max-control-lag-s": args.max_control_lag_s,
    }
    invalid = [name for name, value in positive.items() if not math.isfinite(value) or value <= 0]
    if invalid:
        raise ValueError(f"参数必须为有限正数：{invalid}")
    if args.subscriber_timeout_s < 0 or args.connect_timeout_s < 0:
        raise ValueError("连接和订阅者超时不能小于 0")
    if args.log_every <= 0:
        raise ValueError("--log-every 必须大于 0")
    if args.n_action_steps <= 0:
        raise ValueError("--n-action-steps 必须大于 0")
    start = np.asarray(args.start_joint_positions, dtype=np.float64)
    if start.shape != (7,) or not np.isfinite(start).all():
        raise ValueError("--start-joint-positions 必须包含 7 个有限数值")
    if not math.isfinite(args.start_gripper_position):
        raise ValueError("--start-gripper-position 必须是有限数值")
    if not math.isfinite(args.temporal_ensemble_coeff):
        raise ValueError("--temporal-ensemble-coeff 必须是有限数值")
    if args.move_to_start and not args.send_actions:
        raise ValueError("--move-to-start 会驱动实机，必须同时明确使用 --send-actions")


def configure_temporal_ensemble(
    config: PreTrainedConfig,
    coeff: float,
    requested_n_action_steps: int,
) -> None:
    if config.type != "act":
        raise ValueError(f"本脚本只支持 ACT，checkpoint 类型为 {config.type!r}")
    if requested_n_action_steps > config.chunk_size:
        raise ValueError(
            f"--n-action-steps={requested_n_action_steps} 不能超过 chunk_size={config.chunk_size}"
        )
    if requested_n_action_steps != 1:
        logging.warning(
            "时间集成需要逐帧推理：保留兼容参数 --n-action-steps=%d，但有效值固定为 1",
            requested_n_action_steps,
        )
    config.n_action_steps = 1
    config.temporal_ensemble_coeff = coeff


class ObservationFreshnessWatchdog:
    """Fail if any required ROS stream stops advancing for too long."""

    def __init__(self, timestamps: dict[str, int], now: float) -> None:
        self._timestamps = timestamps.copy()
        self._changed_at = {name: now for name in timestamps}

    def check(self, timestamps: dict[str, int], now: float, timeout_s: float) -> None:
        missing = sorted(set(self._timestamps) - set(timestamps))
        if missing:
            raise RuntimeError(f"observation 缺少时间戳：{missing}")
        for name, value in timestamps.items():
            if self._timestamps.get(name) != value:
                self._timestamps[name] = value
                self._changed_at[name] = now
        stale = [name for name, changed_at in self._changed_at.items() if now - changed_at > timeout_s]
        if stale:
            raise RuntimeError(f"实机 observation 超时未更新（>{timeout_s:.3f}s）：{stale}")


def start_errors(state: np.ndarray, target_joints: np.ndarray, target_gripper: float) -> tuple[float, float]:
    return (
        float(np.max(np.abs(state[:7] - target_joints))),
        float(abs(state[7] - target_gripper)),
    )


def validate_start_state(
    state: np.ndarray,
    target_joints: np.ndarray,
    target_gripper: float,
    arm_tolerance: float,
    gripper_tolerance: float,
    allow_mismatch: bool,
) -> None:
    arm_error, gripper_error = start_errors(state, target_joints, target_gripper)
    if arm_error <= arm_tolerance and gripper_error <= gripper_tolerance:
        logging.info(
            "实机起始姿态检查通过：arm_max_error=%.6f rad, gripper_error=%.6f",
            arm_error,
            gripper_error,
        )
        return
    message = (
        "当前反馈偏离采集起始姿态："
        f"arm_max_error={arm_error:.6f} rad（容差 {arm_tolerance:.6f}），"
        f"gripper_error={gripper_error:.6f}（容差 {gripper_tolerance:.6f}）"
    )
    if not allow_mismatch:
        raise RuntimeError(message + "。请先移动到起始姿态，或明确使用 --move-to-start。")
    logging.warning("%s；已由 --allow-start-mismatch 放行", message)


def log_training_range_difference(
    state: np.ndarray,
    bounds: dict[str, np.ndarray],
) -> None:
    lower = bounds["observation.state.min"]
    upper = bounds["observation.state.max"]
    outside = np.flatnonzero((state < lower) | (state > upper))
    if outside.size:
        details = [
            f"dim{i}={state[i]:.6f} not in [{lower[i]:.6f}, {upper[i]:.6f}]"
            for i in outside
        ]
        logging.warning(
            "起始姿态符合采集复位目标，但位于清洗训练集 min/max 之外：%s",
            "; ".join(details),
        )


def wait_for_subscriber(publisher, topic: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while publisher.get_subscription_count() < 1:
        if time.monotonic() >= deadline:
            raise RuntimeError(f"等待实机订阅者超时：{topic}")
        time.sleep(0.05)


def ensure_exclusive_action_publishers(robot: NeroRos2Deployment) -> None:
    """The deployment node must be the only command publisher on both topics."""
    if robot._node is None:
        raise RuntimeError("ROS 节点尚未创建")
    time.sleep(COMMAND_DISCOVERY_SETTLE_S)
    topics = (
        robot.ARM_CONTROL_TOPIC,
        robot.GRIPPER_CONTROL_TOPIC,
    )
    conflicts: list[str] = []
    for topic in topics:
        publishers = robot._node.get_publishers_info_by_topic(topic)
        external = [
            f"{info.node_namespace.rstrip('/')}/{info.node_name}"
            for info in publishers
            if info.node_name != robot.config.node_name
        ]
        if external:
            conflicts.append(f"{topic}: {external}")
    if conflicts:
        raise RuntimeError(
            "检测到其他控制发布者；请停止 IK、delta-pose 或遥操作节点：" + "; ".join(conflicts)
        )
    move_j_publishers = robot._node.get_publishers_info_by_topic(robot.config.move_j_topic)
    if move_j_publishers:
        sources = [f"{info.node_namespace.rstrip('/')}/{info.node_name}" for info in move_j_publishers]
        raise RuntimeError(f"检测到其他 {robot.config.move_j_topic} 发布者：{sources}")


def publish_start_targets(
    robot: NeroRos2Deployment,
    move_j_publisher,
    target_joints: np.ndarray,
    target_gripper: float,
) -> None:
    if robot._node is None or robot._gripper_control_publisher is None:
        raise RuntimeError("实机控制 publisher 尚未创建")
    stamp = robot._node.get_clock().now().to_msg()

    arm_msg = JointState()
    arm_msg.header.stamp = stamp
    arm_msg.name = list(robot.ARM_JOINT_NAMES)
    arm_msg.position = target_joints.tolist()
    move_j_publisher.publish(arm_msg)

    gripper_msg = JointState()
    gripper_msg.header.stamp = stamp
    gripper_msg.name = [robot.GRIPPER_JOINT_NAME]
    gripper_msg.position = [float(target_gripper)]
    robot._gripper_control_publisher.publish(gripper_msg)


def move_to_start_pose(
    robot: NeroRos2Deployment,
    target_joints: np.ndarray,
    target_gripper: float,
    arm_tolerance: float,
    gripper_tolerance: float,
    stable_time_s: float,
    timeout_s: float,
    republish_s: float,
    observation_timeout_s: float,
) -> None:
    if robot._node is None:
        raise RuntimeError("ROS 节点尚未创建")
    move_j_publisher = robot._node.create_publisher(JointState, robot.config.move_j_topic, 1)
    wait_for_subscriber(move_j_publisher, robot.config.move_j_topic, timeout_s)
    if robot._gripper_control_publisher is None:
        raise RuntimeError("夹爪 publisher 尚未创建")
    wait_for_subscriber(robot._gripper_control_publisher, robot.GRIPPER_CONTROL_TOPIC, timeout_s)

    _, timestamps = robot.get_observation_with_timestamps()
    watchdog = ObservationFreshnessWatchdog(timestamps, time.monotonic())
    deadline = time.monotonic() + timeout_s
    next_publish = 0.0
    stable_since: float | None = None
    last_arm_error = float("inf")
    last_gripper_error = float("inf")
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_publish:
            publish_start_targets(robot, move_j_publisher, target_joints, target_gripper)
            next_publish = now + republish_s
        observation, timestamps = robot.get_observation_with_timestamps()
        watchdog.check(timestamps, now, observation_timeout_s)
        state = observation_state(robot, observation)
        last_arm_error, last_gripper_error = start_errors(state, target_joints, target_gripper)
        if last_arm_error <= arm_tolerance and last_gripper_error <= gripper_tolerance:
            stable_since = stable_since or now
            if now - stable_since >= stable_time_s:
                logging.info(
                    "实机已稳定到达起始姿态：arm_max_error=%.6f rad, gripper_error=%.6f",
                    last_arm_error,
                    last_gripper_error,
                )
                return
        else:
            stable_since = None
        time.sleep(0.01)
    raise TimeoutError(
        "移动到实机起始姿态超时："
        f"arm_max_error={last_arm_error:.6f} rad, gripper_error={last_gripper_error:.6f}"
    )


def hold_current_position(robot: NeroRos2Deployment) -> None:
    try:
        observation = robot.get_observation()
        state = observation_state(robot, observation)
        hold_action = {
            name: float(state[index]) for index, name in enumerate(robot.POSITION_FEATURE_NAMES)
        }
        robot.send_action(hold_action)
        logging.info("退出前已发送一次当前反馈位置作为保持目标")
    except Exception:
        logging.exception("退出前发送保持位置失败，请立即检查实机状态")


def main() -> None:
    args = parse_args()
    validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint}")
    device = resolve_device(args.device)
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    configure_temporal_ensemble(config, args.temporal_ensemble_coeff, args.n_action_steps)
    config.device = device

    robot = NeroRos2Deployment(
        NeroRos2Config(
            id="act_real_deploy",
            node_name=args.node_name,
            reset_mode="disabled",
            connect_timeout_s=args.connect_timeout_s,
        )
    )
    dataset_features = make_dataset_features(robot)
    validate_policy_features(config, dataset_features)
    bounds = load_training_bounds(checkpoint)

    logging.info("加载实机 ACT checkpoint：%s", checkpoint)
    policy = ACTPolicy.from_pretrained(checkpoint, config=config, local_files_only=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    logging.info(
        "模式=%s，device=%s，fps=%.1f，temporal_ensemble_coeff=%g",
        "SEND" if args.send_actions else "DRY-RUN",
        device,
        args.fps,
        args.temporal_ensemble_coeff,
    )

    target_joints = np.asarray(args.start_joint_positions, dtype=np.float64)
    hardware_commanded = False
    robot_connected = False
    try:
        robot.connect()
        robot_connected = True
        ensure_exclusive_action_publishers(robot)

        if args.send_actions:
            wait_for_control_subscribers(robot, args.subscriber_timeout_s)

        if args.move_to_start:
            hardware_commanded = True
            move_to_start_pose(
                robot=robot,
                target_joints=target_joints,
                target_gripper=args.start_gripper_position,
                arm_tolerance=args.start_tolerance_rad,
                gripper_tolerance=args.start_gripper_tolerance,
                stable_time_s=args.start_stable_time_s,
                timeout_s=args.move_timeout_s,
                republish_s=args.move_republish_s,
                observation_timeout_s=args.observation_timeout_s,
            )

        first_observation, first_timestamps = robot.get_observation_with_timestamps()
        first_state = observation_state(robot, first_observation)
        validate_start_state(
            first_state,
            target_joints,
            args.start_gripper_position,
            args.start_tolerance_rad,
            args.start_gripper_tolerance,
            allow_mismatch=(args.allow_start_mismatch or not args.send_actions),
        )
        log_training_range_difference(first_state, bounds)

        first_frame = build_dataset_frame(dataset_features, first_observation, prefix=OBS_STR)
        warmup_action = predict_action(
            observation=first_frame,
            policy=policy,
            device=torch.device(device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=config.use_amp,
            robot_type=robot.robot_type,
        )
        make_robot_action(warmup_action, dataset_features)
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        logging.info("CUDA 预热通过；时间集成历史已清空；未发布预热动作")

        watchdog = ObservationFreshnessWatchdog(first_timestamps, time.monotonic())
        period_s = 1.0 / args.fps
        started = time.perf_counter()
        next_tick = started
        step = 0
        while time.perf_counter() - started < args.duration_s:
            observation, timestamps = robot.get_observation_with_timestamps()
            now_monotonic = time.monotonic()
            watchdog.check(timestamps, now_monotonic, args.observation_timeout_s)
            state = observation_state(robot, observation)
            observation_frame = build_dataset_frame(dataset_features, observation, prefix=OBS_STR)
            action_tensor = predict_action(
                observation=observation_frame,
                policy=policy,
                device=torch.device(device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=config.use_amp,
                robot_type=robot.robot_type,
            )
            raw_action = make_robot_action(action_tensor, dataset_features)
            safe_action, clipped_by = apply_safety_limits(
                robot,
                raw_action,
                state,
                bounds,
                args.max_arm_step_rad,
                args.max_gripper_step,
            )
            if args.send_actions:
                robot.send_action(safe_action)
                hardware_commanded = True

            if step % args.log_every == 0:
                values = [safe_action[name] for name in robot.POSITION_FEATURE_NAMES]
                logging.info(
                    "step=%d mode=%s ensemble_predictions=%d action=%s clipped_max=%.6f",
                    step,
                    "SEND" if args.send_actions else "DRY-RUN",
                    min(step + 1, config.chunk_size),
                    np.array2string(np.asarray(values), precision=4),
                    clipped_by,
                )

            step += 1
            next_tick += period_s
            remaining = next_tick - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            elif -remaining > args.max_control_lag_s:
                raise RuntimeError(
                    f"控制循环累计落后 {-remaining * 1000.0:.1f} ms，超过实机限制，停止发送"
                )
            elif step % args.log_every == 0:
                logging.warning("控制循环落后 %.1f ms", -remaining * 1000.0)
    except KeyboardInterrupt:
        logging.info("收到 Ctrl+C，准备停止实机部署")
    finally:
        if robot_connected and hardware_commanded and args.hold_on_exit:
            hold_current_position(robot)
        robot.disconnect()
        logging.info("实机部署结束；已停止发布并断开 ROS")


if __name__ == "__main__":
    main()
