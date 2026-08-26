from dataclasses import dataclass

from lerobot.teleoperators import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("nero_ros2")
@dataclass
class NeroRos2TeleopConfig(TeleoperatorConfig):
    node_name: str = "lerobot_nero_ros2_teleoperator"
    arm_control_topic: str = "/control/joint_states"
    gripper_control_topic: str = "/sensor/gripper/joint_state"
    gripper_joint_name: str = "sensor_gripper_center_joint"
    # <= 0 表示一直等待两路 action 的首帧。
    connect_timeout_s: float = 5.0
