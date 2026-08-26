#!/usr/bin/env python3

"""Deploy a trained Nero ACT policy through ROS 2, without depth input.

The script deliberately defaults to dry-run mode.  It reuses LeRobot's policy,
preprocessor, postprocessor and inference functions, while the Nero deployment
adapter supplies ROS observations and publishes safe position commands.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import (
    build_dataset_frame,
    dataset_to_policy_features,
    hw_to_dataset_features,
)
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.robots.nero_ros2.config_nero_ros2 import NeroRos2Config
from lerobot.robots.nero_ros2.nero_ros2_deployment import NeroRos2Deployment
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import predict_action


DEFAULT_CHECKPOINT = Path(
    "/home/szy/lerobot-0.4.4-clean/outputs/train/"
    "nero_act_color_fisheye_30hz/checkpoints/100000/pretrained_model"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nero ACT 部署：state + color + fisheye，深度保持运行但不参与推理。"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=10,
        help="每次 ACT 预测后实际执行的动作数；10 在 30 Hz 下约为 0.33 秒。",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="只读取、推理和打印，不发布控制（默认）。",
    )
    mode.add_argument(
        "--send-actions",
        action="store_true",
        help="向 MuJoCo 控制话题发布经过安全限制的动作。",
    )
    parser.add_argument("--max-arm-step-rad", type=float, default=0.05)
    parser.add_argument("--max-gripper-step", type=float, default=0.005)
    parser.add_argument(
        "--allow-ood-start",
        action="store_true",
        help="允许从训练 observation 范围之外启动；首次测试不建议使用。",
    )
    parser.add_argument("--subscriber-timeout-s", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--node-name", default="lerobot_nero_ros2_act_deploy")
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise ValueError("--fps 必须大于 0")
    if args.duration_s <= 0:
        raise ValueError("--duration-s 必须大于 0")
    if args.n_action_steps <= 0:
        raise ValueError("--n-action-steps 必须大于 0")
    if args.max_arm_step_rad <= 0 or args.max_gripper_step <= 0:
        raise ValueError("单步动作限制必须大于 0")
    if args.subscriber_timeout_s < 0:
        raise ValueError("--subscriber-timeout-s 不能小于 0")
    if args.log_every <= 0:
        raise ValueError("--log-every 必须大于 0")


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 无法使用 CUDA")
    return requested


def make_dataset_features(robot: NeroRos2Deployment) -> dict[str, dict]:
    return {
        **hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False),
        **hw_to_dataset_features(robot.action_features, ACTION, use_video=False),
    }


def validate_policy_features(config: PreTrainedConfig, dataset_features: dict[str, dict]) -> None:
    actual = dataset_to_policy_features(dataset_features)
    expected = {**config.input_features, **config.output_features}
    if set(actual) != set(expected):
        raise ValueError(
            "部署特征与 checkpoint 不一致："
            f"缺少={sorted(set(expected) - set(actual))}，"
            f"多余={sorted(set(actual) - set(expected))}"
        )
    mismatched = {
        key: (actual[key].shape, expected[key].shape)
        for key in expected
        if tuple(actual[key].shape) != tuple(expected[key].shape)
    }
    if mismatched:
        raise ValueError(f"部署特征尺寸与 checkpoint 不一致：{mismatched}")


def load_training_bounds(checkpoint: Path) -> dict[str, np.ndarray]:
    candidates = sorted(
        checkpoint.glob("policy_postprocessor_step_*_unnormalizer_processor.safetensors")
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            "无法唯一确定 postprocessor 统计文件："
            f"{[str(path) for path in candidates]}"
        )
    tensors = load_file(candidates[0])
    required = (
        "action.min",
        "action.max",
        "observation.state.min",
        "observation.state.max",
    )
    missing = [key for key in required if key not in tensors]
    if missing:
        raise KeyError(f"训练统计缺少安全范围：{missing}")
    return {key: tensors[key].cpu().numpy().astype(np.float64) for key in required}


def observation_state(robot: NeroRos2Deployment, observation: dict) -> np.ndarray:
    return np.asarray(
        [observation[name] for name in robot.POSITION_FEATURE_NAMES],
        dtype=np.float64,
    )


def check_start_state(
    state: np.ndarray,
    bounds: dict[str, np.ndarray],
    allow_ood_start: bool,
) -> None:
    lower = bounds["observation.state.min"]
    upper = bounds["observation.state.max"]
    outside = np.flatnonzero((state < lower) | (state > upper))
    if outside.size == 0:
        logging.info("初始状态位于训练数据范围内")
        return
    details = [
        f"dim{i}: value={state[i]:.6f}, train=[{lower[i]:.6f}, {upper[i]:.6f}]"
        for i in outside
    ]
    message = "初始状态超出训练分布：" + "; ".join(details)
    if not allow_ood_start:
        raise RuntimeError(message + "。请先调整 MuJoCo 初始姿态，或明确使用 --allow-ood-start。")
    logging.warning(message)


def apply_safety_limits(
    robot: NeroRos2Deployment,
    action: dict[str, float],
    state: np.ndarray,
    bounds: dict[str, np.ndarray],
    max_arm_step_rad: float,
    max_gripper_step: float,
) -> tuple[dict[str, float], float]:
    raw = np.asarray([action[name] for name in robot.POSITION_FEATURE_NAMES], dtype=np.float64)
    if raw.shape != (8,) or not np.isfinite(raw).all():
        raise ValueError(f"模型动作无效：shape={raw.shape}, values={raw.tolist()}")

    clipped = np.clip(raw, bounds["action.min"], bounds["action.max"])
    max_step = np.asarray([max_arm_step_rad] * 7 + [max_gripper_step], dtype=np.float64)
    clipped = np.clip(clipped, state - max_step, state + max_step)
    max_change = float(np.max(np.abs(clipped - raw)))
    safe_action = {
        name: float(clipped[index])
        for index, name in enumerate(robot.POSITION_FEATURE_NAMES)
    }
    return safe_action, max_change


def wait_for_control_subscribers(robot: NeroRos2Deployment, timeout_s: float) -> None:
    arm_publisher = robot._arm_control_publisher
    gripper_publisher = robot._gripper_control_publisher
    if arm_publisher is None or gripper_publisher is None:
        raise RuntimeError("控制 publisher 尚未创建")
    deadline = time.monotonic() + timeout_s
    while arm_publisher.get_subscription_count() < 1 or gripper_publisher.get_subscription_count() < 1:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "MuJoCo 控制订阅者未就绪："
                f"arm={arm_publisher.get_subscription_count()}, "
                f"gripper={gripper_publisher.get_subscription_count()}"
            )
        time.sleep(0.05)
    logging.info("MuJoCo 两个控制订阅者均已就绪")


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
    if config.type != "act":
        raise ValueError(f"本脚本只支持 ACT，checkpoint 类型为 {config.type!r}")
    if args.n_action_steps > config.chunk_size:
        raise ValueError(
            f"--n-action-steps={args.n_action_steps} 不能超过 chunk_size={config.chunk_size}"
        )
    config.device = device
    config.n_action_steps = args.n_action_steps

    robot_config = NeroRos2Config(
        id="act_deploy",
        node_name=args.node_name,
        connect_timeout_s=args.connect_timeout_s,
    )
    robot = NeroRos2Deployment(robot_config)
    dataset_features = make_dataset_features(robot)
    validate_policy_features(config, dataset_features)
    bounds = load_training_bounds(checkpoint)

    logging.info("加载 ACT checkpoint：%s", checkpoint)
    policy = ACTPolicy.from_pretrained(
        checkpoint,
        config=config,
        local_files_only=True,
    )
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    logging.info(
        "模式=%s，device=%s，fps=%.1f，n_action_steps=%d",
        "SEND" if args.send_actions else "DRY-RUN",
        device,
        args.fps,
        args.n_action_steps,
    )

    try:
        robot.connect()
        first_observation = robot.get_observation()
        check_start_state(
            observation_state(robot, first_observation),
            bounds,
            allow_ood_start=(args.allow_ood_start or not args.send_actions),
        )

        # Warm up CUDA and validate the complete inference path without publishing.
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
        logging.info("首帧纯推理检查通过，未发布动作")

        if args.send_actions:
            wait_for_control_subscribers(robot, args.subscriber_timeout_s)

        period_s = 1.0 / args.fps
        started = time.perf_counter()
        next_tick = started
        step = 0
        while time.perf_counter() - started < args.duration_s:
            observation = robot.get_observation()
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

            if step % args.log_every == 0:
                values = [safe_action[name] for name in robot.POSITION_FEATURE_NAMES]
                logging.info(
                    "step=%d mode=%s action=%s clipped_max=%.6f",
                    step,
                    "SEND" if args.send_actions else "DRY-RUN",
                    np.array2string(np.asarray(values), precision=4),
                    clipped_by,
                )

            step += 1
            next_tick += period_s
            remaining = next_tick - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            elif step % args.log_every == 0:
                logging.warning("控制循环落后 %.1f ms", -remaining * 1000.0)
    except KeyboardInterrupt:
        logging.info("收到 Ctrl+C，停止发布")
    finally:
        robot.disconnect()
        logging.info("部署结束；已停止发布并断开 ROS")


if __name__ == "__main__":
    main()
