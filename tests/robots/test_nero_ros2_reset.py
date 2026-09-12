from types import SimpleNamespace
import threading
from unittest.mock import Mock

import pytest

from lerobot.robots.nero_ros2.config_nero_ros2 import NeroRos2Config
from lerobot.robots.nero_ros2.nero_ros2 import NeroRos2


def test_reset_on_start_defaults_are_safe_for_real_hardware() -> None:
    assert NeroRos2Config(id="sim", reset_mode="simulation").reset_on_start is True
    assert NeroRos2Config(id="real", reset_mode="real").reset_on_start is True
    assert NeroRos2Config(id="off", reset_mode="disabled").reset_on_start is False


def test_real_reset_requires_seven_joint_targets() -> None:
    with pytest.raises(ValueError, match="exactly 7"):
        NeroRos2Config(id="real", reset_mode="real", reset_joint_positions=[0.0])


def make_real_reset_robot() -> NeroRos2:
    robot = NeroRos2.__new__(NeroRos2)
    robot._connected = True
    robot.config = SimpleNamespace(real_reset_timeout_s=20.0, reset_timeout_s=5.0)
    robot._set_control_enabled = Mock()
    robot._delta_pose_is_active = Mock(return_value=True)
    robot._toggle_teleop = Mock()
    robot._wait_for_delta_pose_to_stop = Mock()
    robot._wait_for_action_command_to_stop = Mock()
    robot._publish_real_reset_pose = Mock(return_value=123)
    robot._wait_for_real_reset_pose = Mock()
    robot._wait_for_safe_resumed_action = Mock()
    return robot


def test_real_reset_gates_motion_and_rebases_before_resuming() -> None:
    robot = make_real_reset_robot()

    robot._reset_real()

    gate_calls = robot._set_control_enabled.call_args_list
    assert [args.args[0] for args in gate_calls] == [False, True, False, True]
    assert len({args.args[1] for args in gate_calls}) == 1
    assert robot._toggle_teleop.call_count == 4
    robot._wait_for_delta_pose_to_stop.assert_called_once()
    robot._wait_for_action_command_to_stop.assert_called_once()
    robot._wait_for_real_reset_pose.assert_called_once_with(
        123, robot._wait_for_real_reset_pose.call_args.args[1]
    )
    robot._wait_for_safe_resumed_action.assert_called_once()


def test_real_reset_closes_gate_when_motion_fails() -> None:
    robot = make_real_reset_robot()
    robot._wait_for_real_reset_pose.side_effect = TimeoutError("motion timeout")

    with pytest.raises(TimeoutError, match="motion timeout"):
        robot._reset_real()

    assert [args.args[0] for args in robot._set_control_enabled.call_args_list] == [False, True, False]


def test_wait_for_action_command_requires_quiet_period_after_late_ik(monkeypatch) -> None:
    robot = NeroRos2.__new__(NeroRos2)
    robot.config = SimpleNamespace(teleop_activity_window_s=0.15)
    robot._cache_lock = threading.Lock()
    robot._action_command_count = 10
    clock = [0.0]
    late_command_sent = [False]

    def fake_monotonic() -> float:
        return clock[0]

    def fake_sleep(duration: float) -> None:
        before = clock[0]
        clock[0] += duration
        if before < 0.3 <= clock[0] and not late_command_sent[0]:
            robot._action_command_count += 1
            late_command_sent[0] = True

    monkeypatch.setattr("lerobot.robots.nero_ros2.nero_ros2.time.monotonic", fake_monotonic)
    monkeypatch.setattr("lerobot.robots.nero_ros2.nero_ros2.time.sleep", fake_sleep)

    robot._wait_for_action_command_to_stop(deadline=2.0)

    assert late_command_sent[0]
    assert clock[0] >= 0.3 + robot.RESET_IK_QUIET_TIME_S


def test_wait_for_real_reset_republishes_and_reports_each_joint(monkeypatch) -> None:
    robot = NeroRos2.__new__(NeroRos2)
    robot.config = SimpleNamespace(
        reset_joint_positions=[0.0] * 7,
        reset_position_tolerance_rad=0.03,
        reset_stable_time_s=0.3,
    )
    robot._cache_lock = threading.Lock()
    robot._arm_positions = (0.0, 0.0, 0.17, 0.08, 0.0, 0.0, 0.0)
    robot._arm_receive_monotonic_ns = 2
    robot._publish_real_reset_pose = Mock(return_value=3)
    clock = [0.0]

    monkeypatch.setattr(
        "lerobot.robots.nero_ros2.nero_ros2.time.monotonic", lambda: clock[0]
    )
    monkeypatch.setattr(
        "lerobot.robots.nero_ros2.nero_ros2.time.sleep",
        lambda duration: clock.__setitem__(0, clock[0] + duration),
    )

    with pytest.raises(TimeoutError, match=r"joint3: 当前=0\.1700.*joint4: 当前=0\.0800"):
        robot._wait_for_real_reset_pose(published_at_ns=1, deadline=4.1)

    assert robot._publish_real_reset_pose.call_count == 2
