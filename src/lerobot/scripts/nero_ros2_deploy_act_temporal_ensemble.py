#!/usr/bin/env python3

"""Deploy a Nero ACT policy through ROS 2 with ACT temporal ensembling.

This is the temporal-ensemble counterpart of ``nero_ros2_deploy_act.py``.
It keeps that script's command-line interface and safety behavior, but queries
ACT on every control step and combines all overlapping action predictions for
the current step.  Depth continues running in ROS and is excluded from policy
inference by ``NeroRos2Deployment``.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.robots.nero_ros2.config_nero_ros2 import NeroRos2Config
from lerobot.robots.nero_ros2.nero_ros2_deployment import NeroRos2Deployment
from lerobot.scripts.nero_ros2_deploy_act import (
    DEFAULT_CHECKPOINT,
    apply_safety_limits,
    check_start_state,
    load_training_bounds,
    make_dataset_features,
    observation_state,
    resolve_device,
    validate_policy_features,
    wait_for_control_subscribers,
)
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action


DEFAULT_TEMPORAL_ENSEMBLE_COEFF = 0.01


def parse_args() -> argparse.Namespace:
    """Parse the original deployment interface plus the ensemble coefficient."""
    parser = argparse.ArgumentParser(
        description="Nero ACT 时间集成部署：state + color + fisheye，深度保持运行但不参与推理。"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=10,
        help=(
            "兼容原部署程序的参数；时间集成要求每帧推理，因此本程序始终使用 1，"
            "传入的值不会改变执行方式。"
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=DEFAULT_TEMPORAL_ENSEMBLE_COEFF,
        help=(
            "ACT 时间集成指数权重系数，默认 0.01（原始 ACT 取值）；"
            "0 表示等权，正值更偏重较早预测。"
        ),
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
    if not math.isfinite(args.temporal_ensemble_coeff):
        raise ValueError("--temporal-ensemble-coeff 必须是有限数值")
    if args.max_arm_step_rad <= 0 or args.max_gripper_step <= 0:
        raise ValueError("单步动作限制必须大于 0")
    if args.subscriber_timeout_s < 0:
        raise ValueError("--subscriber-timeout-s 不能小于 0")
    if args.log_every <= 0:
        raise ValueError("--log-every 必须大于 0")


def enable_temporal_ensemble(
    config: PreTrainedConfig,
    coeff: float,
    requested_n_action_steps: int,
) -> None:
    """Enable ACT's online temporal ensembler before constructing the policy."""
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
    enable_temporal_ensemble(config, args.temporal_ensemble_coeff, args.n_action_steps)
    config.device = device

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
        "模式=%s，device=%s，fps=%.1f，temporal_ensemble_coeff=%g，effective_n_action_steps=1",
        "SEND" if args.send_actions else "DRY-RUN",
        device,
        args.fps,
        args.temporal_ensemble_coeff,
    )

    try:
        robot.connect()
        first_observation = robot.get_observation()
        check_start_state(
            observation_state(robot, first_observation),
            bounds,
            allow_ood_start=(args.allow_ood_start or not args.send_actions),
        )

        # Warm up the complete path, then clear the warm-up prediction so the
        # first real action starts with a clean temporal-ensemble history.
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
        logging.info("首帧纯推理检查通过，时间集成历史已清空，未发布动作")

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

            # With temporal_ensemble_coeff set, select_action predicts a full
            # chunk every step and ACTTemporalEnsembler returns the weighted
            # average of all predictions that target this control timestep.
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
            elif step % args.log_every == 0:
                logging.warning("控制循环落后 %.1f ms", -remaining * 1000.0)
    except KeyboardInterrupt:
        logging.info("收到 Ctrl+C，停止发布")
    finally:
        robot.disconnect()
        logging.info("部署结束；已停止发布并断开 ROS")


if __name__ == "__main__":
    main()
