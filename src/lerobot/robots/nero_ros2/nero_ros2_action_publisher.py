#!/usr/bin/env python3

"""Nero ROS 2 adapter that publishes policy actions to the existing control topics.

The original :mod:`nero_ros2` implementation is intentionally left unchanged.
This subclass only adds ROS publishers and implements ``send_action``.
"""

from __future__ import annotations

import math

from sensor_msgs.msg import JointState

from lerobot.processor import RobotAction

from .nero_ros2 import NeroRos2


class NeroRos2ActionPublisher(NeroRos2):
    """Split an eight-value LeRobot action into arm and gripper commands."""

    ARM_CONTROL_TOPIC = "/control/joint_states"
    GRIPPER_CONTROL_TOPIC = "/sensor/gripper/joint_state"
    GRIPPER_JOINT_NAME = "sensor_gripper_center_joint"

    def __init__(self, config):
        super().__init__(config)
        self._arm_control_publisher = None
        self._gripper_control_publisher = None

    def connect(self, calibrate: bool = True) -> None:
        """Connect the original ROS adapter, then create the command publishers."""
        super().connect(calibrate=calibrate)
        if self._node is None:
            raise RuntimeError("NeroRos2 ROS node was not created")

        self._arm_control_publisher = self._node.create_publisher(
            JointState,
            self.ARM_CONTROL_TOPIC,
            10,
        )
        self._gripper_control_publisher = self._node.create_publisher(
            JointState,
            self.GRIPPER_CONTROL_TOPIC,
            10,
        )

    def send_action(self, action: RobotAction) -> RobotAction:
        """Publish seven arm positions and one gripper opening as JointState."""
        if not self._connected or self._node is None:
            raise RuntimeError("NeroRos2ActionPublisher is not connected")
        if self._arm_control_publisher is None or self._gripper_control_publisher is None:
            raise RuntimeError("NeroRos2ActionPublisher command publishers are not ready")

        missing = [name for name in self.POSITION_FEATURE_NAMES if name not in action]
        if missing:
            raise ValueError(f"action is missing fields: {missing}")

        accepted_action = {name: float(action[name]) for name in self.POSITION_FEATURE_NAMES}
        invalid = [name for name, value in accepted_action.items() if not math.isfinite(value)]
        if invalid:
            raise ValueError(f"action contains non-finite values: {invalid}")

        stamp = self._node.get_clock().now().to_msg()

        arm_msg = JointState()
        arm_msg.header.stamp = stamp
        arm_msg.name = list(self.ARM_JOINT_NAMES)
        arm_msg.position = [accepted_action[f"{name}.pos"] for name in self.ARM_JOINT_NAMES]

        gripper_msg = JointState()
        gripper_msg.header.stamp = stamp
        gripper_msg.name = [self.GRIPPER_JOINT_NAME]
        gripper_msg.position = [accepted_action["gripper.pos"]]

        self._arm_control_publisher.publish(arm_msg)
        self._gripper_control_publisher.publish(gripper_msg)
        return accepted_action

    def disconnect(self) -> None:
        """Drop publisher references and let the original adapter stop ROS."""
        self._arm_control_publisher = None
        self._gripper_control_publisher = None
        super().disconnect()


__all__ = ["NeroRos2ActionPublisher"]
