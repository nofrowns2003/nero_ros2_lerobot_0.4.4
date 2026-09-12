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
from geometry_msgs.msg import PoseStamped
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import SetBool, Trigger

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
    # A solve can finish a few hundred milliseconds after /delta_pose stops.
    # Require a comfortably longer quiet period before allowing reset commands
    # through the shared hardware gate.
    RESET_IK_QUIET_TIME_S = 1.0
    # move_j is a one-shot ROS message which becomes a short burst of CAN
    # commands. Re-issue the same target occasionally until feedback confirms
    # arrival, so a missed command cannot strand the arm indefinitely.
    RESET_REPUBLISH_INTERVAL_S = 2.0

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
        self._control_enable_client: Any | None = None
        self._teleop_trigger_client: Any | None = None
        self._move_j_publisher: Any | None = None
        self._bridge = CvBridge()
        self._connected = False
        self._cache_lock = threading.Lock()
        self._arm_positions: tuple[float, ...] | None = None
        self._gripper_position: float | None = None
        # Image callbacks publish newly-created arrays into the write buffer. The
        # recording thread swaps the two dictionaries and only reads from the read
        # buffer. Arrays are never modified after publication, so handing a read
        # buffer array to an asynchronous dataset writer is safe without copying it.
        self._image_write_buffer: dict[str, np.ndarray | None] = {
            name: None for name in self.IMAGE_FEATURE_NAMES
        }
        self._image_read_buffer: dict[str, np.ndarray | None] = {
            name: None for name in self.IMAGE_FEATURE_NAMES
        }
        self._image_write_dirty = False
        # ROS header 时间戳统一保存成 int64 语义的纳秒整数，避免浮点精度损失。
        self._topic_timestamps_ns: dict[str, int | None] = {name: None for name in self.TIMESTAMP_NAMES}
        # 这里保存最近一次 get_observation() 对应的时间戳快照。
        self._last_observation_timestamps_ns: dict[str, int] | None = None
        self._last_observation_snapshot: tuple[RobotObservation, dict[str, int]] | None = None
        self._arm_receive_monotonic_ns = 0
        self._delta_pose_count = 0
        self._action_command_count = 0
        self._last_action_command: tuple[float, ...] | None = None
        self._last_action_command_monotonic_ns = 0

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
            self._arm_receive_monotonic_ns = time.monotonic_ns()

    def _gripper_state_callback(self, msg: Gripper) -> None:
        with self._cache_lock:
            self._gripper_position = float(msg.distance)
            self._topic_timestamps_ns["gripper"] = self._stamp_to_ns(msg)

    def _make_image_callback(self, key: str, encoding: str) -> Callable[[Image], None]:
        def callback(msg: Image) -> None:
            try:
                image = np.asarray(self._bridge.imgmsg_to_cv2(msg, desired_encoding=encoding))
                image = self._depth_to_rgb(image, msg.encoding) if key == "depth" else self._check_rgb(image, key)
                # Conversion/contiguity checks may copy a complete image. Keep that
                # work outside the cache lock so get_observation() is never held up
                # by image processing.
                cached_image = np.ascontiguousarray(image, dtype=np.uint8)
            except (CvBridgeError, TypeError, ValueError) as exc:
                logger.warning("忽略无效的 %s 图像：%s", key, exc)
                return
            with self._cache_lock:
                self._image_write_buffer[key] = cached_image
                self._topic_timestamps_ns[key] = self._stamp_to_ns(msg)
                self._image_write_dirty = True

        return callback

    def _delta_pose_callback(self, msg: PoseStamped) -> None:
        del msg
        with self._cache_lock:
            self._delta_pose_count += 1

    def _action_command_callback(self, msg: JointState) -> None:
        if len(msg.name) != len(msg.position):
            return
        position_map = {name: float(value) for name, value in zip(msg.name, msg.position)}
        if any(name not in position_map for name in self.ARM_JOINT_NAMES):
            return
        with self._cache_lock:
            self._action_command_count += 1
            self._last_action_command = tuple(position_map[name] for name in self.ARM_JOINT_NAMES)
            self._last_action_command_monotonic_ns = time.monotonic_ns()

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
            missing.extend(
                image_topics[key] for key, value in self._image_write_buffer.items() if value is None
            )
            return missing

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self._connected or self._context is not None:
            raise RuntimeError("NeroRos2 已经连接")
        with self._cache_lock:
            self._arm_positions = None
            self._gripper_position = None
            self._image_write_buffer = {name: None for name in self.IMAGE_FEATURE_NAMES}
            self._image_read_buffer = {name: None for name in self.IMAGE_FEATURE_NAMES}
            self._image_write_dirty = False
            self._topic_timestamps_ns = {name: None for name in self.TIMESTAMP_NAMES}
            self._last_observation_timestamps_ns = None
            self._last_observation_snapshot = None
            self._arm_receive_monotonic_ns = 0
            self._delta_pose_count = 0
            self._action_command_count = 0
            self._last_action_command = None
            self._last_action_command_monotonic_ns = 0
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
            if self.config.reset_mode == "simulation":
                self._reset_client = self._node.create_client(Trigger, self.config.reset_service)
            elif self.config.reset_mode == "real":
                self._control_enable_client = self._node.create_client(
                    SetBool, self.config.control_enable_service
                )
                self._teleop_trigger_client = self._node.create_client(
                    Trigger, self.config.teleop_trigger_service
                )
                self._move_j_publisher = self._node.create_publisher(
                    JointState, self.config.move_j_topic, 1
                )
                self._subscriptions.extend(
                    [
                        self._node.create_subscription(
                            PoseStamped,
                            self.config.delta_pose_topic,
                            self._delta_pose_callback,
                            qos,
                        ),
                        self._node.create_subscription(
                            JointState,
                            self.config.action_command_topic,
                            self._action_command_callback,
                            qos,
                        ),
                    ]
                )
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
            # Startup may wait for the first complete observation. Publish it once
            # here so every subsequent real-time read has a non-blocking fallback.
            with self._cache_lock:
                self._publish_observation_snapshot_locked()
        except Exception:
            self.disconnect()
            raise

    def get_observation(self) -> RobotObservation:
        observation, _ = self._take_observation_snapshot()
        return observation

    def _publish_observation_snapshot_locked(self) -> tuple[RobotObservation, dict[str, int]]:
        """Publish the current write-side cache; caller must hold ``_cache_lock``."""
        if (
            self._arm_positions is None
            or self._gripper_position is None
            or any(value is None for value in self._image_write_buffer.values())
        ):
            raise RuntimeError("observation 缓存未就绪")
        if any(value is None for value in self._topic_timestamps_ns.values()):
            raise RuntimeError("话题时间戳缓存未就绪")

        if self._image_write_dirty:
            self._image_read_buffer, self._image_write_buffer = (
                self._image_write_buffer,
                self._image_read_buffer,
            )
            # The old read dictionary becomes the next write buffer. Copy only
            # three ndarray references so an update to one camera cannot make the
            # other two cameras fall back to an older frame on the next swap.
            self._image_write_buffer.update(self._image_read_buffer)
            self._image_write_dirty = False

        positions = self._arm_positions
        gripper = self._gripper_position
        images = {
            key: value for key, value in self._image_read_buffer.items() if value is not None
        }
        timestamps = {
            key: int(value) for key, value in self._topic_timestamps_ns.items() if value is not None
        }
        observation: RobotObservation = {
            **{f"{name}.pos": positions[index] for index, name in enumerate(self.ARM_JOINT_NAMES)},
            "gripper.pos": gripper,
            **images,
        }
        self._last_observation_timestamps_ns = timestamps
        self._last_observation_snapshot = (observation, timestamps)
        return observation.copy(), timestamps.copy()

    def _take_observation_snapshot(self) -> tuple[RobotObservation, dict[str, int]]:
        """立即返回最新完整快照；cache 正在更新时返回上一份快照。"""
        if not self._connected:
            raise RuntimeError("NeroRos2 未连接")

        if self._cache_lock.acquire(blocking=False):
            try:
                return self._publish_observation_snapshot_locked()
            finally:
                self._cache_lock.release()

        # A callback owns the cache for a few pointer/scalar assignments. Do not
        # wait for it: the previously published immutable snapshot is still valid.
        snapshot = self._last_observation_snapshot
        if snapshot is None:
            raise RuntimeError("observation 完整快照尚未发布")
        observation, timestamps = snapshot
        return observation.copy(), timestamps.copy()

    def get_observation_timestamps(self) -> dict[str, int]:
        """返回最近一次 get_observation() 固化的五路时间戳，单位为纳秒。"""
        if not self._connected:
            raise RuntimeError("NeroRos2 未连接")
        timestamps = self._last_observation_timestamps_ns
        if timestamps is None:
            raise RuntimeError("请先调用 get_observation()")
        return timestamps.copy()

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

    def _wait_for_client(self, client: Any, service_name: str, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not client.wait_for_service(timeout_sec=remaining):
            raise TimeoutError(f"等待服务 {service_name} 超时")

    def _wait_for_future(self, future: Any, description: str, deadline: float) -> Any:
        while not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{description}超时")
            time.sleep(0.01)
        response = future.result()
        if response is None:
            raise RuntimeError(f"{description}没有收到响应")
        return response

    def _set_control_enabled(self, enabled: bool, deadline: float) -> None:
        if self._control_enable_client is None:
            raise RuntimeError("实机控制门服务客户端未初始化")
        self._wait_for_client(
            self._control_enable_client, self.config.control_enable_service, deadline
        )
        response = self._wait_for_future(
            self._control_enable_client.call_async(SetBool.Request(data=enabled)),
            "切换实机控制门",
            deadline,
        )
        if not response.success:
            raise RuntimeError(f"切换实机控制门失败：{response.message}")
        logger.info("实机控制门已%s", "打开" if enabled else "关闭")

    def _toggle_teleop(self, deadline: float) -> None:
        if self._teleop_trigger_client is None:
            raise RuntimeError("遥操作开关服务客户端未初始化")
        self._wait_for_client(
            self._teleop_trigger_client, self.config.teleop_trigger_service, deadline
        )
        response = self._wait_for_future(
            self._teleop_trigger_client.call_async(Trigger.Request()),
            "切换遥操作状态",
            deadline,
        )
        if not response.success:
            raise RuntimeError(f"切换遥操作状态失败：{response.message}")

    def _delta_pose_is_active(self) -> bool:
        with self._cache_lock:
            count_before = self._delta_pose_count
        time.sleep(self.config.teleop_activity_window_s)
        with self._cache_lock:
            return self._delta_pose_count > count_before

    def _wait_for_delta_pose_to_stop(self, deadline: float) -> None:
        while time.monotonic() < deadline:
            if not self._delta_pose_is_active():
                return
        raise TimeoutError("暂停遥操作后 /delta_pose 仍在持续更新")

    def _action_command_is_active(self) -> bool:
        with self._cache_lock:
            count_before = self._action_command_count
        time.sleep(self.config.teleop_activity_window_s)
        with self._cache_lock:
            return self._action_command_count > count_before

    def _wait_for_action_command_to_stop(self, deadline: float) -> None:
        """Require continuous IK-output silence before opening the reset gate.

        A single short sampling window is insufficient: an IK solve may still
        be running during that window and publish its result afterwards. Such a
        late command would overwrite the one-shot move_j reset target.
        """
        with self._cache_lock:
            last_count = self._action_command_count
        quiet_since = time.monotonic()
        poll_interval_s = min(self.config.teleop_activity_window_s, 0.05)
        while time.monotonic() < deadline:
            time.sleep(min(poll_interval_s, max(deadline - time.monotonic(), 0.0)))
            now = time.monotonic()
            with self._cache_lock:
                current_count = self._action_command_count
            if current_count != last_count:
                last_count = current_count
                quiet_since = now
                continue
            if now - quiet_since >= self.RESET_IK_QUIET_TIME_S:
                logger.info("IK 动作输出已连续静默 %.1fs", self.RESET_IK_QUIET_TIME_S)
                return
        raise TimeoutError("暂停遥操作后 IK 的 /control/joint_states 仍在持续更新")

    def _publish_real_reset_pose(self, deadline: float) -> int:
        if self._move_j_publisher is None or self._node is None:
            raise RuntimeError("实机 move_j publisher 未初始化")
        while self._move_j_publisher.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"等待 {self.config.move_j_topic} 订阅者超时")
            time.sleep(0.01)

        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = list(self.ARM_JOINT_NAMES)
        msg.position = [float(value) for value in self.config.reset_joint_positions]
        published_at_ns = time.monotonic_ns()
        self._move_j_publisher.publish(msg)
        logger.info("已发布实机复位目标：%s", msg.position)
        return published_at_ns

    def _wait_for_real_reset_pose(self, published_at_ns: int, deadline: float) -> None:
        target = tuple(float(value) for value in self.config.reset_joint_positions)
        stable_since: float | None = None
        last_error = float("inf")
        last_positions: tuple[float, ...] | None = None
        next_republish_at = time.monotonic() + self.RESET_REPUBLISH_INTERVAL_S
        while time.monotonic() < deadline:
            with self._cache_lock:
                positions = self._arm_positions
                received_at_ns = self._arm_receive_monotonic_ns
            if positions is not None and received_at_ns > published_at_ns:
                last_positions = positions
                last_error = max(abs(actual - expected) for actual, expected in zip(positions, target))
                now = time.monotonic()
                if last_error <= self.config.reset_position_tolerance_rad:
                    if stable_since is None:
                        stable_since = now
                    elif now - stable_since >= self.config.reset_stable_time_s:
                        logger.info("实机已到达复位位置，最大关节误差 %.4f rad", last_error)
                        return
                else:
                    stable_since = None
                if now >= next_republish_at:
                    logger.warning(
                        "实机尚未到达复位位置，最大关节误差 %.4f rad；重新发布复位目标",
                        last_error,
                    )
                    self._publish_real_reset_pose(deadline)
                    next_republish_at = now + self.RESET_REPUBLISH_INTERVAL_S
            time.sleep(0.01)
        if last_positions is None:
            detail = "未收到复位命令发布后的新关节反馈"
        else:
            detail = ", ".join(
                f"{name}: 当前={actual:.4f}, 目标={expected:.4f}, 误差={abs(actual - expected):.4f}"
                for name, actual, expected in zip(self.ARM_JOINT_NAMES, last_positions, target)
            )
        raise TimeoutError(
            f"等待实机到达复位位置超时，最大关节误差 {last_error:.4f} rad；{detail}"
        )

    def _wait_for_safe_resumed_action(self, enabled_at_ns: int, deadline: float) -> None:
        last_error = float("inf")
        while time.monotonic() < deadline:
            with self._cache_lock:
                command = self._last_action_command
                command_at_ns = self._last_action_command_monotonic_ns
                positions = self._arm_positions
            if command is not None and positions is not None and command_at_ns > enabled_at_ns:
                last_error = max(abs(actual - target) for actual, target in zip(positions, command))
                if last_error <= self.config.resume_action_tolerance_rad:
                    logger.info("遥操作重对零检查通过，恢复目标误差 %.4f rad", last_error)
                    return
            time.sleep(0.01)
        raise RuntimeError(
            "遥操作重对零后的首个 IK 目标不安全或未收到："
            f"最大关节误差 {last_error:.4f} rad，控制门保持关闭"
        )

    def _reset_simulation(self) -> None:
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

    def _reset_real(self) -> None:
        """Safely return the real arm to a configured pose and rebase Pika teleoperation."""
        if not self._connected:
            raise RuntimeError("NeroRos2 未连接，无法复位实机")
        deadline = time.monotonic() + self.config.real_reset_timeout_s
        gate_closed = False
        try:
            # Block every /control/* callback before inspecting or toggling teleoperation.
            self._set_control_enabled(False, deadline)
            gate_closed = True

            if self._delta_pose_is_active():
                self._toggle_teleop(deadline)
                self._wait_for_delta_pose_to_stop(deadline)
                logger.info("遥操作已暂停")
            else:
                logger.info("遥操作原本已暂停")

            # A final IK solve may still be running after /delta_pose becomes
            # quiet. Keep the hardware gate closed until its output is quiet too,
            # otherwise that late result can overwrite the one-shot move_j reset.
            logger.info(
                "等待 IK 动作输出连续静默 %.1fs 后再发布复位目标",
                self.RESET_IK_QUIET_TIME_S,
            )
            self._wait_for_action_command_to_stop(deadline)

            # Teleoperation is now quiet, so the gate can be opened solely for move_j.
            self._set_control_enabled(True, deadline)
            gate_closed = False
            published_at_ns = self._publish_real_reset_pose(deadline)
            self._wait_for_real_reset_pose(published_at_ns, deadline)

            # Rebase with the hardware gate closed.  The existing Pika service is a
            # toggle, so ON -> OFF captures the current TCP, and the final ON captures
            # the current handle pose against that TCP.  Any intermediate IK output is
            # rejected by the driver while the gate is closed.
            self._set_control_enabled(False, deadline)
            gate_closed = True
            self._toggle_teleop(deadline)
            self._toggle_teleop(deadline)
            enabled_at_ns = time.monotonic_ns()
            self._toggle_teleop(deadline)
            self._wait_for_safe_resumed_action(enabled_at_ns, deadline)

            self._set_control_enabled(True, deadline)
            gate_closed = False
            logger.info("实机复位完成，遥操作已基于当前位置重新对零")
        except Exception:
            if not gate_closed:
                try:
                    emergency_deadline = time.monotonic() + min(2.0, self.config.reset_timeout_s)
                    self._set_control_enabled(False, emergency_deadline)
                    gate_closed = True
                except Exception:
                    logger.exception("实机复位失败后无法关闭控制门")
            if gate_closed:
                logger.error("实机复位失败；为安全起见控制门保持关闭")
            else:
                logger.critical("实机复位失败且无法确认控制门状态，请立即急停并检查驱动")
            raise

    def reset(self) -> None:
        if self.config.reset_mode == "simulation":
            self._reset_simulation()
        elif self.config.reset_mode == "real":
            self._reset_real()
        else:
            logger.info("reset_mode=disabled，忽略 episode 复位请求")

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
        self._control_enable_client = None
        self._teleop_trigger_client = None
        self._move_j_publisher = None
        self._executor = None
        self._spin_thread = None
        self._node = None
        self._context = None
