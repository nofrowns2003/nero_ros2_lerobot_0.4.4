import threading

import numpy as np

from lerobot.robots.nero_ros2.nero_ros2 import NeroRos2


def make_observation_robot() -> NeroRos2:
    robot = NeroRos2.__new__(NeroRos2)
    robot._connected = True
    robot._cache_lock = threading.Lock()
    robot._arm_positions = tuple(float(index) for index in range(7))
    robot._gripper_position = 0.5
    robot._image_write_buffer = {
        name: np.full((4, 5, 3), index, dtype=np.uint8)
        for index, name in enumerate(robot.IMAGE_FEATURE_NAMES)
    }
    robot._image_read_buffer = {name: None for name in robot.IMAGE_FEATURE_NAMES}
    robot._image_write_dirty = True
    robot._topic_timestamps_ns = {
        name: index + 1 for index, name in enumerate(robot.TIMESTAMP_NAMES)
    }
    robot._last_observation_timestamps_ns = None
    robot._last_observation_snapshot = None
    return robot


def test_observation_buffer_swaps_image_references_without_copying() -> None:
    robot = make_observation_robot()
    written_images = robot._image_write_buffer.copy()

    observation, timestamps = robot.get_observation_with_timestamps()

    assert timestamps == robot._topic_timestamps_ns
    for name, image in written_images.items():
        assert observation[name] is image


def test_partial_camera_update_keeps_other_latest_images() -> None:
    robot = make_observation_robot()
    first_observation, _ = robot.get_observation_with_timestamps()
    new_color = np.full((4, 5, 3), 99, dtype=np.uint8)

    with robot._cache_lock:
        robot._image_write_buffer["color"] = new_color
        robot._topic_timestamps_ns["color"] = 100
        robot._image_write_dirty = True

    second_observation, timestamps = robot.get_observation_with_timestamps()

    assert second_observation["color"] is new_color
    assert second_observation["depth"] is first_observation["depth"]
    assert second_observation["fisheye"] is first_observation["fisheye"]
    assert timestamps["color"] == 100


def test_observation_read_does_not_wait_when_callback_owns_cache_lock() -> None:
    robot = make_observation_robot()
    published_observation, published_timestamps = robot.get_observation_with_timestamps()

    robot._cache_lock.acquire()
    try:
        observation, timestamps = robot.get_observation_with_timestamps()
    finally:
        robot._cache_lock.release()

    assert timestamps == published_timestamps
    for name in robot.IMAGE_FEATURE_NAMES:
        assert observation[name] is published_observation[name]
