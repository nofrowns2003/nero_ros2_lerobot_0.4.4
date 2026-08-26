#!/usr/bin/env python3

"""Nero ROS 2 deployment adapter that leaves depth running but ignores it for inference.

The original robot adapter and action-publisher adapter are intentionally left
unchanged.  The camera may continue publishing depth normally, but this adapter
does not decode, cache, wait for, or expose depth to LeRobot.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sensor_msgs.msg import Image

from .nero_ros2_action_publisher import NeroRos2ActionPublisher


class NeroRos2Deployment(NeroRos2ActionPublisher):
    """Use state, color and fisheye for inference while retaining ROS depth data."""

    IMAGE_FEATURE_NAMES = ("color", "fisheye")
    TIMESTAMP_NAMES = ("joint_state", "gripper", "color", "fisheye")
    DEPTH_FEATURE_NAME = "depth"

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        """Describe exactly the features used by the depth-free ACT policy."""
        image_shape = (self.config.image_height, self.config.image_width, 3)
        return {
            **{name: float for name in self.POSITION_FEATURE_NAMES},
            **{name: image_shape for name in self.IMAGE_FEATURE_NAMES},
        }

    def _make_image_callback(self, key: str, encoding: str) -> Callable[[Image], None]:
        """Discard depth messages before conversion while preserving other cameras."""
        if key == self.DEPTH_FEATURE_NAME:
            return self._ignore_depth_callback
        return super()._make_image_callback(key, encoding)

    @staticmethod
    def _ignore_depth_callback(msg: Any) -> None:
        del msg


__all__ = ["NeroRos2Deployment"]
