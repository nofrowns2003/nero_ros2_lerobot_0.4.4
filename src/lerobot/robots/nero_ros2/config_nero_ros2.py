#!/usr/bin/env python3

from dataclasses import dataclass

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

    image_width: int = 640
    image_height: int = 480
    # 16UC1 深度的 [0, depth_max_mm] 会映射为三通道 uint8 图像。
    depth_max_mm: float = 5000.0
    # <= 0 表示 connect() 一直等待所有 observation 的首帧。
    connect_timeout_s: float = 5.0
    reset_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image_width and image_height must be positive")
        if self.depth_max_mm <= 0:
            raise ValueError("depth_max_mm must be positive")
        if not self.reset_service:
            raise ValueError("reset_service cannot be empty")
        if self.reset_timeout_s <= 0:
            raise ValueError("reset_timeout_s must be positive")
