#!/usr/bin/env python3

import math
from dataclasses import dataclass, field

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("nero_ros2")
@dataclass
class NeroRos2Config(RobotConfig):
    node_name: str = "lerobot_nero_ros2_robot"

    joint_state_topic: str = "/feedback/joint_states"
    gripper_state_topic: str = "/gripper/gripper/data"
    color_image_topic: str = "/gripper/camera/color/image_raw"
    depth_image_topic: str = "/gripper/camera/aligned_depth_to_color/image_raw"
    fisheye_image_topic: str = "/gripper/camera_fisheye/color/image_raw"
    reset_service: str = "/reset_episode"

    # Episode reset backend:
    # - "simulation": call reset_service (the historical MuJoCo behavior)
    # - "real": pause/rebase Pika teleoperation and command a configurable pose
    # - "disabled": never reset automatically or in response to F8
    reset_mode: str = "simulation"
    # By default every enabled reset backend also resets before episode 0.
    # Set this explicitly to false only when the operator has already placed the
    # robot safely and intentionally wants to skip the first automatic reset.
    reset_on_start: bool | None = None

    # Real-arm reset interface.  The default pose matches the MuJoCo "home"
    # keyframe and can be overridden from the lerobot-record command line.
    reset_joint_positions: list[float] = field(
        default_factory=lambda: [0.0, -1.74, 0.0, 2.14, 0.0, 0.0, 1.57]
    )
    move_j_topic: str = "/control/move_j"
    action_command_topic: str = "/control/joint_states"
    delta_pose_topic: str = "/delta_pose"
    control_enable_service: str = "/control_enable"
    teleop_trigger_service: str = "/teleop_trigger"

    image_width: int = 640
    image_height: int = 480
    # 16UC1 深度的 [0, depth_max_mm] 会映射为三通道 uint8 图像。
    depth_max_mm: float = 5000.0
    # <= 0 表示 connect() 一直等待所有 observation 的首帧。
    connect_timeout_s: float = 5.0
    reset_timeout_s: float = 5.0
    real_reset_timeout_s: float = 20.0
    reset_position_tolerance_rad: float = 0.03
    reset_stable_time_s: float = 0.3
    # After reset/tracking has settled, keep reading ROS callbacks without
    # writing dataset frames and show a terminal countdown before frame 0.
    record_countdown_s: int = 3
    teleop_activity_window_s: float = 0.15
    resume_action_tolerance_rad: float = 0.08

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image_width and image_height must be positive")
        if self.depth_max_mm <= 0:
            raise ValueError("depth_max_mm must be positive")
        if self.reset_mode not in {"simulation", "real", "disabled"}:
            raise ValueError("reset_mode must be one of: simulation, real, disabled")
        if self.reset_on_start is None:
            self.reset_on_start = self.reset_mode != "disabled"
        if self.reset_mode == "simulation" and not self.reset_service:
            raise ValueError("reset_service cannot be empty in simulation reset mode")
        if self.reset_timeout_s <= 0:
            raise ValueError("reset_timeout_s must be positive")
        if len(self.reset_joint_positions) != 7:
            raise ValueError("reset_joint_positions must contain exactly 7 joint values")
        if not all(math.isfinite(float(value)) for value in self.reset_joint_positions):
            raise ValueError("reset_joint_positions must contain only finite values")
        if self.real_reset_timeout_s <= 0:
            raise ValueError("real_reset_timeout_s must be positive")
        if self.reset_position_tolerance_rad <= 0:
            raise ValueError("reset_position_tolerance_rad must be positive")
        if self.reset_stable_time_s <= 0:
            raise ValueError("reset_stable_time_s must be positive")
        if self.record_countdown_s < 0:
            raise ValueError("record_countdown_s must be non-negative")
        if self.teleop_activity_window_s <= 0:
            raise ValueError("teleop_activity_window_s must be positive")
        if self.resume_action_tolerance_rad <= 0:
            raise ValueError("resume_action_tolerance_rad must be positive")
        if self.reset_mode == "real":
            required_names = {
                "move_j_topic": self.move_j_topic,
                "action_command_topic": self.action_command_topic,
                "delta_pose_topic": self.delta_pose_topic,
                "control_enable_service": self.control_enable_service,
                "teleop_trigger_service": self.teleop_trigger_service,
            }
            empty = [name for name, value in required_names.items() if not value]
            if empty:
                raise ValueError(f"real reset interface cannot be empty: {empty}")
