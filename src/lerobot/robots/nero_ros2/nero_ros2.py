#!/usr/bin/env python3

"""使用 ROS 2 最新消息缓存的 LeRobot Nero 机器人适配器。"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from data_msgs.msg import Gripper
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots import Robot

from .config_nero_ros2 import NeroRos2Config

logger = logging.getLogger(__name__)


class NeroRos2(Robot):
    """缓存五路话题的最新值，不做时间戳同步。"""

    config_class = NeroRos2Config
    name = "nero_ros2"
    ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
    POSITION_FEATURE_NAMES = tuple(f"{name}.pos" for name in ARM_JOINT_NAMES) + ("gripper.pos",)
    IMAGE_FEATURE_NAMES = ("color", "depth", "fisheye")
    TIMESTAMP_NAMES = ("joint_state", "gripper", "color", "depth", "fisheye")

    def __init__(self, config: NeroRos2Config):
        super().__init__(config)
        self.config = config
        # lerobot-record 用 cameras 数量创建图像写入线程；ROS 图像无需 Camera 实例。
        self.cameras: dict[str, None] = {name: None for name in self.IMAGE_FEATURE_NAMES}
        self._context: Context | None = None
        self._node: Any | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._subscriptions: list[Any] = []
        self._reset_client: Any | None = None
        self._bridge = CvBridge()
        self._connected = False
        self._cache_lock = threading.Lock()
        self._arm_positions: tuple[float, ...] | None = None
        self._gripper_position: float | None = None
        self._images: dict[str, np.ndarray | None] = {name: None for name in self.IMAGE_FEATURE_NAMES}
        # ROS header 时间戳统一保存成 int64 语义的纳秒整数，避免浮点精度损失。
        self._topic_timestamps_ns: dict[str, int | None] = {name: None for name in self.TIMESTAMP_NAMES}
        # 这里保存最近一次 get_observation() 对应的时间戳快照。
        self._last_observation_timestamps_ns: dict[str, int] | None = None

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        image_shape = (self.config.image_height, self.config.image_width, 3)
        return {
            **{name: float for name in self.POSITION_FEATURE_NAMES},
            **{name: image_shape for name in self.IMAGE_FEATURE_NAMES},
        }

    @property
    def action_features(self) -> dict[str, type]:
        # action 与机器人状态使用同一组关节定义：7 个机械臂关节角 + 1 个夹爪角。
        return {name: float for name in self.POSITION_FEATURE_NAMES}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @staticmethod
    def _stamp_to_ns(msg: Any) -> int:
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _arm_state_callback(self, msg: JointState) -> None:
        if len(msg.name) != len(msg.position):
            logger.warning("忽略无效关节消息：name 和 position 长度不一致")
            return
        position_map = {name: float(value) for name, value in zip(msg.name, msg.position)}
        missing = [name for name in self.ARM_JOINT_NAMES if name not in position_map]
        if missing:
            logger.warning("忽略缺少关节的反馈：%s", missing)
            return
        with self._cache_lock:
            self._arm_positions = tuple(position_map[name] for name in self.ARM_JOINT_NAMES)
            self._topic_timestamps_ns["joint_state"] = self._stamp_to_ns(msg)

    def _gripper_state_callback(self, msg: Gripper) -> None:
        with self._cache_lock:
            self._gripper_position = float(msg.distance)
            self._topic_timestamps_ns["gripper"] = self._stamp_to_ns(msg)

    def _make_image_callback(self, key: str, encoding: str) -> Callable[[Image], None]:
        def callback(msg: Image) -> None:
            try:
                image = np.asarray(self._bridge.imgmsg_to_cv2(msg, desired_encoding=encoding))
                image = self._depth_to_rgb(image, msg.encoding) if key == "depth" else self._check_rgb(image, key)
            except (CvBridgeError, TypeError, ValueError) as exc:
                logger.warning("忽略无效的 %s 图像：%s", key, exc)
                return
            with self._cache_lock:
                self._images[key] = np.ascontiguousarray(image, dtype=np.uint8)
                self._topic_timestamps_ns[key] = self._stamp_to_ns(msg)

        return callback

    def _check_rgb(self, image: np.ndarray, key: str) -> np.ndarray:
        expected = (self.config.image_height, self.config.image_width, 3)
        if image.shape != expected or image.dtype != np.uint8:
            raise ValueError(f"{key}: expected uint8 {expected}, got {image.dtype} {image.shape}")
        return image

    def _depth_to_rgb(self, depth: np.ndarray, source_encoding: str) -> np.ndarray:
        expected = (self.config.image_height, self.config.image_width)
        if depth.shape != expected:
            raise ValueError(f"depth: expected {expected}, got {depth.shape}")
        depth_mm = depth.astype(np.float32, copy=False)
        if source_encoding.lower() == "32fc1":
            depth_mm = depth_mm * 1000.0
        depth_mm = np.nan_to_num(depth_mm, nan=0.0, posinf=self.config.depth_max_mm, neginf=0.0)
        gray = np.clip(depth_mm * 255.0 / self.config.depth_max_mm, 0.0, 255.0).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=2)

    def _missing_topics(self) -> list[str]:
        with self._cache_lock:
            missing = []
            if self._arm_positions is None:
                missing.append(self.config.joint_state_topic)
            if self._gripper_position is None:
                missing.append(self.config.gripper_state_topic)
            image_topics = {
                "color": self.config.color_image_topic,
                "depth": self.config.depth_image_topic,
                "fisheye": self.config.fisheye_image_topic,
            }
            missing.extend(image_topics[key] for key, value in self._images.items() if value is None)
            return missing

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self._connected or self._context is not None:
            raise RuntimeError("NeroRos2 已经连接")
        with self._cache_lock:
            self._arm_positions = None
            self._gripper_position = None
            self._images = {name: None for name in self.IMAGE_FEATURE_NAMES}
            self._topic_timestamps_ns = {name: None for name in self.TIMESTAMP_NAMES}
            self._last_observation_timestamps_ns = None
        try:
            self._context = Context()
            rclpy.init(args=None, context=self._context)
            self._node = rclpy.create_node(self.config.node_name, context=self._context)
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self._subscriptions = [
                self._node.create_subscription(JointState, self.config.joint_state_topic, self._arm_state_callback, qos),
                self._node.create_subscription(Gripper, self.config.gripper_state_topic, self._gripper_state_callback, qos),
                self._node.create_subscription(Image, self.config.color_image_topic, self._make_image_callback("color", "rgb8"), qos),
                self._node.create_subscription(Image, self.config.depth_image_topic, self._make_image_callback("depth", "passthrough"), qos),
                self._node.create_subscription(Image, self.config.fisheye_image_topic, self._make_image_callback("fisheye", "rgb8"), qos),
            ]
            self._reset_client = self._node.create_client(Trigger, self.config.reset_service)
            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self._node)
            self._spin_thread = threading.Thread(target=self._executor.spin, name="nero_ros2_executor", daemon=True)
            self._spin_thread.start()
            self._connected = True
            deadline = None if self.config.connect_timeout_s <= 0 else time.monotonic() + self.config.connect_timeout_s
            while missing := self._missing_topics():
                if deadline is not None and time.monotonic() >= deadline:
                    raise RuntimeError(f"等待 ROS observation 超时：{', '.join(missing)}")
                time.sleep(0.01)
        except Exception:
            self.disconnect()
            raise

    def get_observation(self) -> RobotObservation:
        observation, _ = self._take_observation_snapshot()
        return observation

    def _take_observation_snapshot(self) -> tuple[RobotObservation, dict[str, int]]:
        """原子取得 observation 和它所对应的五路 ROS 时间戳。"""
        if not self._connected:
            raise RuntimeError("NeroRos2 未连接")
        with self._cache_lock:
            if self._arm_positions is None or self._gripper_position is None or any(v is None for v in self._images.values()):
                raise RuntimeError("observation 缓存未就绪")
            if any(value is None for value in self._topic_timestamps_ns.values()):
                raise RuntimeError("话题时间戳缓存未就绪")
            positions = self._arm_positions
            gripper = self._gripper_position
            images = {key: value.copy() for key, value in self._images.items() if value is not None}
            timestamps = {key: int(value) for key, value in self._topic_timestamps_ns.items() if value is not None}
            self._last_observation_timestamps_ns = timestamps.copy()
        observation = {
            **{f"{name}.pos": positions[index] for index, name in enumerate(self.ARM_JOINT_NAMES)},
            "gripper.pos": gripper,
            **images,
        }
        return observation, timestamps

    def get_observation_timestamps(self) -> dict[str, int]:
        """返回最近一次 get_observation() 固化的五路时间戳，单位为纳秒。"""
        if not self._connected:
            raise RuntimeError("NeroRos2 未连接")
        with self._cache_lock:
            if self._last_observation_timestamps_ns is None:
                raise RuntimeError("请先调用 get_observation()")
            return self._last_observation_timestamps_ns.copy()

    def get_observation_with_timestamps(self) -> tuple[RobotObservation, dict[str, int]]:
        """一次原子调用同时返回 observation 和对应的话题时间戳。"""
        return self._take_observation_snapshot()

    def send_action(self, action: RobotAction) -> RobotAction:
        """校验并返回 action，但暂时不向任何 ROS 话题发布。"""
        if not self._connected:
            raise RuntimeError("NeroRos2 未连接")

        missing = [name for name in self.POSITION_FEATURE_NAMES if name not in action]
        if missing:
            raise ValueError(f"action 缺少字段：{missing}")

        accepted_action = {name: float(action[name]) for name in self.POSITION_FEATURE_NAMES}
        logger.debug("action 发布尚未实现，仅接收并返回 action：%s", accepted_action)
        return accepted_action

    def reset(self) -> None:
        """Reset MuJoCo through ROS and wait until every observation topic is fresh."""
        if not self._connected or self._reset_client is None:
            raise RuntimeError("NeroRos2 未连接，无法重置仿真")

        timeout_s = self.config.reset_timeout_s
        if not self._reset_client.wait_for_service(timeout_sec=timeout_s):
            raise TimeoutError(f"等待重置服务 {self.config.reset_service} 超时（{timeout_s:.1f}s）")

        with self._cache_lock:
            timestamps_before = self._topic_timestamps_ns.copy()

        future = self._reset_client.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout_s
        while not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"调用重置服务 {self.config.reset_service} 超时（{timeout_s:.1f}s）")
            time.sleep(0.01)

        response = future.result()
        if response is None or not response.success:
            message = "没有收到有效响应" if response is None else response.message
            raise RuntimeError(f"MuJoCo 重置失败：{message}")

        while True:
            with self._cache_lock:
                timestamps_after = self._topic_timestamps_ns.copy()
            all_fresh = all(
                before is not None and after is not None and after > before
                for name in self.TIMESTAMP_NAMES
                for before, after in [(timestamps_before[name], timestamps_after[name])]
            )
            if all_fresh:
                break
            if time.monotonic() >= deadline:
                stale = [
                    name
                    for name in self.TIMESTAMP_NAMES
                    if timestamps_before[name] is None
                    or timestamps_after[name] is None
                    or timestamps_after[name] <= timestamps_before[name]
                ]
                raise TimeoutError(f"MuJoCo 已重置，但等待新 observation 超时：{stale}")
            time.sleep(0.01)

        logger.info("MuJoCo episode 已重置：%s", response.message)

    def disconnect(self) -> None:
        self._connected = False
        if self._executor is not None:
            try:
                self._executor.shutdown(timeout_sec=1.0)
            except Exception:
                logger.exception("ROS executor 关闭失败")
        if self._spin_thread is not None and self._spin_thread is not threading.current_thread():
            self._spin_thread.join(timeout=1.0)
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                logger.exception("ROS node 销毁失败")
        if self._context is not None:
            try:
                rclpy.shutdown(context=self._context)
            except Exception:
                logger.exception("ROS context 关闭失败")
        self._subscriptions = []
        self._reset_client = None
        self._executor = None
        self._spin_thread = None
        self._node = None
        self._context = None
