"""从独立 ROS 2 子进程获取 Nero 遥操作 action。"""

import logging
import multiprocessing as mp
import queue
import threading
import time
import traceback
from typing import Any

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from lerobot.processor import RobotAction
from lerobot.teleoperators import Teleoperator

from .config_nero_ros2 import NeroRos2TeleopConfig

logger = logging.getLogger(__name__)

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
GRIPPER_INDEX = 7


def _stamp_to_ns(msg: JointState) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def _action_ros_worker(
    node_name: str,
    arm_topic: str,
    gripper_topic: str,
    gripper_joint_name: str,
    values: Any,
    timestamps_ns: Any,
    receive_times_ns: Any,
    receive_counts: Any,
    shared_lock: Any,
    arm_ready: Any,
    gripper_ready: Any,
    stop_event: Any,
    error_queue: Any,
) -> None:
    """ROS-only child process. Callbacks overwrite a small shared-memory snapshot."""
    context = None
    arm_node = gripper_node = None
    arm_executor = gripper_executor = None
    arm_thread = gripper_thread = None
    subscriptions = []

    def arm_callback(msg: JointState) -> None:
        if len(msg.name) != len(msg.position):
            return
        position_map = {name: float(value) for name, value in zip(msg.name, msg.position)}
        if any(name not in position_map for name in ARM_JOINT_NAMES):
            return
        with shared_lock:
            for index, name in enumerate(ARM_JOINT_NAMES):
                values[index] = position_map[name]
            timestamps_ns[0] = _stamp_to_ns(msg)
            receive_times_ns[0] = time.monotonic_ns()
            receive_counts[0] += 1
        arm_ready.set()

    def gripper_callback(msg: JointState) -> None:
        if len(msg.name) != len(msg.position):
            return
        position_map = {name: float(value) for name, value in zip(msg.name, msg.position)}
        if gripper_joint_name not in position_map:
            return
        with shared_lock:
            values[GRIPPER_INDEX] = position_map[gripper_joint_name]
            timestamps_ns[1] = _stamp_to_ns(msg)
            receive_times_ns[1] = time.monotonic_ns()
            receive_counts[1] += 1
        gripper_ready.set()

    try:
        context = Context()
        rclpy.init(args=None, context=context)
        arm_node = rclpy.create_node(f"{node_name}_arm", context=context)
        gripper_node = rclpy.create_node(f"{node_name}_gripper", context=context)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        subscriptions = [
            arm_node.create_subscription(JointState, arm_topic, arm_callback, qos),
            gripper_node.create_subscription(JointState, gripper_topic, gripper_callback, qos),
        ]
        arm_executor = SingleThreadedExecutor(context=context)
        gripper_executor = SingleThreadedExecutor(context=context)
        arm_executor.add_node(arm_node)
        gripper_executor.add_node(gripper_node)
        arm_thread = threading.Thread(target=arm_executor.spin, name="nero_action_arm", daemon=True)
        gripper_thread = threading.Thread(
            target=gripper_executor.spin, name="nero_action_gripper", daemon=True
        )
        arm_thread.start()
        gripper_thread.start()
        while not stop_event.wait(0.05):
            pass
    except Exception:
        error_queue.put(traceback.format_exc())
    finally:
        # Keep references alive until both executors have stopped.
        _ = subscriptions
        for executor in (arm_executor, gripper_executor):
            if executor is not None:
                try:
                    executor.shutdown(timeout_sec=1.0)
                except Exception:
                    pass
        for thread in (arm_thread, gripper_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=1.0)
        for node in (arm_node, gripper_node):
            if node is not None:
                try:
                    node.destroy_node()
                except Exception:
                    pass
        if context is not None:
            try:
                rclpy.shutdown(context=context)
            except Exception:
                pass


class NeroRos2Teleop(Teleoperator):
    """通过共享内存读取独立 ROS 子进程缓存的最新 action。"""

    config_class = NeroRos2TeleopConfig
    name = "nero_ros2"
    ARM_JOINT_NAMES = ARM_JOINT_NAMES
    ACTION_FEATURE_NAMES = tuple(f"{name}.pos" for name in ARM_JOINT_NAMES) + ("gripper.pos",)
    TIMESTAMP_NAMES = ("joint_state", "gripper")

    def __init__(self, config: NeroRos2TeleopConfig):
        super().__init__(config)
        self.config = config
        # spawn avoids inheriting an already initialized DDS/rclpy context.
        self._mp = mp.get_context("spawn")
        self._values = self._mp.Array("d", 8, lock=False)
        self._timestamps_ns = self._mp.Array("q", 2, lock=False)
        self._receive_times_ns = self._mp.Array("q", 2, lock=False)
        self._receive_counts = self._mp.Array("q", 2, lock=False)
        self._shared_lock = self._mp.Lock()
        self._arm_ready = self._mp.Event()
        self._gripper_ready = self._mp.Event()
        self._stop_event = self._mp.Event()
        self._error_queue = self._mp.Queue()
        self._process: mp.Process | None = None
        self._connected = False
        self._snapshot_lock = threading.Lock()
        self._last_action_timestamps_ns: dict[str, int] | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return {name: float for name in self.ACTION_FEATURE_NAMES}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected and self._process is not None and self._process.is_alive()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _worker_error(self) -> str | None:
        try:
            return self._error_queue.get_nowait()
        except queue.Empty:
            return None

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self._process is not None:
            raise RuntimeError("NeroRos2Teleop 已经连接")
        # Reconnect 时不让上一个 worker 留下的错误消息污染本次启动。
        while self._worker_error() is not None:
            pass
        with self._shared_lock:
            for index in range(8):
                self._values[index] = 0.0
            for index in range(2):
                self._timestamps_ns[index] = 0
                self._receive_times_ns[index] = 0
                self._receive_counts[index] = 0
        self._arm_ready.clear()
        self._gripper_ready.clear()
        self._stop_event.clear()
        with self._snapshot_lock:
            self._last_action_timestamps_ns = None

        self._process = self._mp.Process(
            target=_action_ros_worker,
            name="nero_ros2_action_process",
            daemon=True,
            args=(
                self.config.node_name,
                self.config.arm_control_topic,
                self.config.gripper_control_topic,
                self.config.gripper_joint_name,
                self._values,
                self._timestamps_ns,
                self._receive_times_ns,
                self._receive_counts,
                self._shared_lock,
                self._arm_ready,
                self._gripper_ready,
                self._stop_event,
                self._error_queue,
            ),
        )
        self._process.start()
        deadline = (
            None
            if self.config.connect_timeout_s <= 0
            else time.monotonic() + self.config.connect_timeout_s
        )
        try:
            while not (self._arm_ready.is_set() and self._gripper_ready.is_set()):
                error = self._worker_error()
                if error is not None:
                    raise RuntimeError(f"Action ROS 子进程启动失败：\n{error}")
                if not self._process.is_alive():
                    raise RuntimeError("Action ROS 子进程意外退出")
                if deadline is not None and time.monotonic() >= deadline:
                    missing = []
                    if not self._arm_ready.is_set():
                        missing.append(self.config.arm_control_topic)
                    if not self._gripper_ready.is_set():
                        missing.append(self.config.gripper_control_topic)
                    raise RuntimeError(f"等待 ROS action 超时：{', '.join(missing)}")
                time.sleep(0.01)
        except Exception:
            self.disconnect()
            raise
        self._connected = True

    def _take_action_snapshot(self) -> tuple[RobotAction, dict[str, int]]:
        if not self.is_connected:
            error = self._worker_error()
            suffix = f"：\n{error}" if error else ""
            raise RuntimeError(f"NeroRos2Teleop 未连接或子进程已退出{suffix}")
        with self._shared_lock:
            values = [float(self._values[index]) for index in range(8)]
            timestamps = {
                "joint_state": int(self._timestamps_ns[0]),
                "gripper": int(self._timestamps_ns[1]),
            }
        with self._snapshot_lock:
            self._last_action_timestamps_ns = timestamps.copy()
        action: RobotAction = {
            **{f"{name}.pos": values[index] for index, name in enumerate(self.ARM_JOINT_NAMES)},
            "gripper.pos": values[GRIPPER_INDEX],
        }
        return action, timestamps

    def get_action(self) -> RobotAction:
        action, _ = self._take_action_snapshot()
        return action

    def get_action_timestamps(self) -> dict[str, int]:
        """返回最近一次 get_action() 对应的时间戳快照，单位为纳秒。"""
        if not self.is_connected:
            raise RuntimeError("NeroRos2Teleop 未连接")
        with self._snapshot_lock:
            if self._last_action_timestamps_ns is None:
                raise RuntimeError("请先调用 get_action()")
            return self._last_action_timestamps_ns.copy()

    def get_action_with_timestamps(self) -> tuple[RobotAction, dict[str, int]]:
        return self._take_action_snapshot()

    def get_receive_stats(self) -> dict[str, dict[str, int]]:
        """返回子进程 callback 接收计数和最近本机接收时刻，便于诊断频率。"""
        with self._shared_lock:
            return {
                "joint_state": {
                    "count": int(self._receive_counts[0]),
                    "last_receive_monotonic_ns": int(self._receive_times_ns[0]),
                },
                "gripper": {
                    "count": int(self._receive_counts[1]),
                    "last_receive_monotonic_ns": int(self._receive_times_ns[1]),
                },
            }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if not self.is_connected:
            raise RuntimeError("NeroRos2Teleop 未连接")
        if feedback:
            logger.debug("feedback 尚未实现，忽略：%s", feedback)

    def disconnect(self) -> None:
        self._connected = False
        process = self._process
        if process is None:
            return
        self._stop_event.set()
        process.join(timeout=3.0)
        if process.is_alive():
            logger.warning("Action ROS 子进程未及时退出，将终止该子进程")
            process.terminate()
            process.join(timeout=1.0)
        self._process = None
