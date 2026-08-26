#!/usr/bin/env python

"""Align a Nero ROS 2 LeRobot dataset using the recorded ROS header timestamps.

RGB is the reference clock. Depth and fisheye frames are matched one-to-one,
while state values use nearest-neighbour matching and actions use causal
zero-order hold (the newest action whose timestamp is not later than RGB).
"""

from __future__ import annotations

import argparse
import bisect
import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES


RGB_KEY = "observation.images.color"
DEPTH_KEY = "observation.images.depth"
FISHEYE_KEY = "observation.images.fisheye"
STATE_KEY = "observation.state"
ACTION_KEY = "action"

ROS_COLUMNS = (
    "episode_index",
    "frame_index",
    "lerobot_timestamp",
    "observation_joint_state",
    "observation_gripper",
    "observation_color",
    "observation_depth",
    "observation_fisheye",
    "action_joint_state",
    "action_gripper",
)

ROS_SCHEMA = pa.schema(
    [
        ("episode_index", pa.int64()),
        ("frame_index", pa.int64()),
        ("lerobot_timestamp", pa.float64()),
        ("observation_joint_state", pa.int64()),
        ("observation_gripper", pa.int64()),
        ("observation_color", pa.int64()),
        ("observation_depth", pa.int64()),
        ("observation_fisheye", pa.int64()),
        ("action_joint_state", pa.int64()),
        ("action_gripper", pa.int64()),
    ]
)


@dataclass(frozen=True)
class TimestampedSource:
    timestamp_ns: int
    source_frame_index: int


@dataclass(frozen=True)
class AlignedFrame:
    color: TimestampedSource
    depth: TimestampedSource
    fisheye: TimestampedSource
    observation_joint: TimestampedSource
    observation_gripper: TimestampedSource
    action_joint: TimestampedSource
    action_gripper: TimestampedSource


class SequentialVideoReader:
    """Read selected monotonically increasing frame numbers without buffering a video."""

    def __init__(self, path: Path):
        self.path = path
        self.container = av.open(str(path))
        streams = list(self.container.streams.video)
        if not streams:
            self.container.close()
            raise RuntimeError(f"视频中没有 video stream：{path}")
        self.stream = streams[0]
        self.stream.thread_type = "AUTO"
        self._frames = self.container.decode(self.stream)
        self._current_index = -1

    def read(self, target_index: int) -> np.ndarray:
        if target_index <= self._current_index:
            raise RuntimeError(
                f"视频源帧索引必须严格递增：{target_index} <= {self._current_index} ({self.path})"
            )
        while self._current_index < target_index:
            try:
                frame = next(self._frames)
            except StopIteration as exc:
                raise RuntimeError(
                    f"视频提前结束，无法读取第 {target_index} 帧：{self.path}"
                ) from exc
            self._current_index += 1
        return frame.to_ndarray(format="rgb24")

    def close(self) -> None:
        self.container.close()

    def __enter__(self) -> SequentialVideoReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class RosTimestampStore:
    """Load new per-episode timestamp files and the legacy combined file."""

    def __init__(self, dataset_root: Path):
        self.directory = dataset_root / "ros_timestamps"
        self.legacy_path = self.directory / "ros2_timestamp.parquet"
        self._legacy_table: pa.Table | None = None

    def load_episode(self, episode_index: int) -> dict[str, list[Any]]:
        episode_path = self.directory / f"episode_{episode_index:06d}.parquet"
        if episode_path.exists():
            table = pq.read_table(episode_path)
        elif self.legacy_path.exists():
            if self._legacy_table is None:
                self._legacy_table = pq.read_table(self.legacy_path)
            table = self._legacy_table.filter(
                pc.equal(self._legacy_table["episode_index"], episode_index)
            )
        else:
            raise FileNotFoundError(
                f"找不到 episode {episode_index} 的 ROS 时间戳文件：{self.directory}"
            )

        missing = set(ROS_COLUMNS) - set(table.column_names)
        if missing:
            raise RuntimeError(f"ROS 时间戳文件缺少列：{sorted(missing)}")
        rows = table.select(ROS_COLUMNS).sort_by("frame_index").to_pydict()
        episode_values = {int(value) for value in rows["episode_index"]}
        if episode_values != {episode_index}:
            raise RuntimeError(
                f"episode {episode_index} 时间戳文件中包含其他 episode：{episode_values}"
            )
        return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="以 RGB ROS 时间戳为基准，重新对齐 Nero ROS 2 LeRobot 数据集。"
    )
    parser.add_argument("--source-root", type=Path, required=True, help="原始 LeRobot 数据集目录")
    parser.add_argument("--output-root", type=Path, help="新数据集目录（不允许已存在）")
    parser.add_argument("--output-repo-id", help="新数据集 repo_id")
    episode_group = parser.add_mutually_exclusive_group()
    episode_group.add_argument(
        "--episode",
        default="all",
        metavar="INDEX|all",
        help="要处理的一轮，例如 --episode=1；--episode=all 处理全部（默认）",
    )
    episode_group.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        help=argparse.SUPPRESS,  # 兼容旧版多 episode 参数。
    )
    parser.add_argument(
        "--max-depth-delta-ms",
        type=float,
        default=20.0,
        help="Depth 与 RGB 允许的最大绝对时间差；<=0 表示不限制",
    )
    parser.add_argument(
        "--max-fisheye-delta-ms",
        type=float,
        default=20.0,
        help="Fisheye 与 RGB 允许的最大绝对时间差；<=0 表示不限制",
    )
    parser.add_argument(
        "--max-state-delta-ms",
        type=float,
        default=50.0,
        help="反馈关节/夹爪与 RGB 允许的最大绝对时间差；<=0 表示不限制",
    )
    parser.add_argument(
        "--max-action-age-ms",
        type=float,
        default=0.0,
        help="Action 允许比 RGB 旧的最大时间；<=0 表示不限制",
    )
    parser.add_argument("--vcodec", default="libsvtav1", help="输出视频编码器")
    parser.add_argument("--encoder-threads", type=int, help="每路视频编码线程数")
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=0,
        help="仅保留每个 episode 前 N 个匹配，用于小规模验证；0 表示全部",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只计算匹配关系和报告，不解码视频、不写数据集",
    )
    return parser.parse_args()


def milliseconds_to_ns(value: float) -> int | None:
    return None if value <= 0 else int(value * 1_000_000)


def select_episode_indices(
    episode: str,
    legacy_episodes: list[int] | None,
    episode_count: int,
) -> list[int]:
    if legacy_episodes is not None:
        selected = legacy_episodes
    elif episode.lower() == "all":
        selected = list(range(episode_count))
    else:
        try:
            selected = [int(episode)]
        except ValueError as exc:
            raise ValueError(
                f"--episode 只能是非负整数或 all，当前为：{episode!r}"
            ) from exc
    if len(set(selected)) != len(selected):
        raise ValueError(f"episode 不能重复：{selected}")
    invalid = {index for index in selected if index < 0 or index >= episode_count}
    if invalid:
        raise ValueError(
            f"episode 不存在：{sorted(invalid)}；当前数据集共 {episode_count} 轮"
        )
    return selected


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def normalize_features(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Restore tuple shapes lost when dataset features were serialized as JSON."""
    normalized = copy.deepcopy(features)
    for feature in normalized.values():
        if isinstance(feature.get("shape"), list):
            feature["shape"] = tuple(feature["shape"])
    return normalized


def load_episode_metadata(root: Path) -> list[dict[str, Any]]:
    paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"找不到 episode metadata：{root / 'meta' / 'episodes'}")
    table = pa.concat_tables([pq.read_table(path) for path in paths], promote_options="default")
    rows = table.sort_by("episode_index").to_pylist()
    indices = [int(row["episode_index"]) for row in rows]
    if indices != list(range(len(indices))):
        raise RuntimeError(f"episode_index 必须从 0 连续递增，当前为：{indices}")
    return rows


def load_tasks(root: Path) -> dict[int, str]:
    table = pq.read_table(root / "meta" / "tasks.parquet")
    name_column = next(
        (name for name in table.column_names if name != "task_index"),
        None,
    )
    if name_column is None:
        raise RuntimeError("tasks.parquet 中找不到 task 文本列")
    return {
        int(task_index): str(task_name)
        for task_index, task_name in zip(
            table["task_index"].to_pylist(), table[name_column].to_pylist(), strict=True
        )
    }


def load_episode_data(
    root: Path,
    info: dict[str, Any],
    episode_meta: dict[str, Any],
) -> dict[str, list[Any]]:
    path = root / info["data_path"].format(
        chunk_index=int(episode_meta["data/chunk_index"]),
        file_index=int(episode_meta["data/file_index"]),
    )
    table = pq.read_table(path)
    table = table.filter(pc.equal(table["episode_index"], int(episode_meta["episode_index"])))
    table = table.sort_by("frame_index")
    return table.to_pydict()


def make_stream(rows: dict[str, list[Any]], column: str) -> list[TimestampedSource]:
    """Create a strictly ordered stream, collapsing repeated latest-cache samples."""
    stream: list[TimestampedSource] = []
    last_timestamp = -1
    for frame_index, raw_timestamp in zip(rows["frame_index"], rows[column], strict=True):
        timestamp = int(raw_timestamp)
        if timestamp <= 0:
            raise RuntimeError(f"{column} 包含无效时间戳：frame={frame_index}, value={timestamp}")
        if timestamp < last_timestamp:
            raise RuntimeError(
                f"{column} 时间戳发生倒退：frame={frame_index}, "
                f"{timestamp} < {last_timestamp}"
            )
        if timestamp == last_timestamp:
            continue
        stream.append(TimestampedSource(timestamp, int(frame_index)))
        last_timestamp = timestamp
    if not stream:
        raise RuntimeError(f"{column} 没有可用时间戳")
    return stream


def nearest_index(
    stream: list[TimestampedSource],
    timestamp_ns: int,
    start_index: int = 0,
) -> int:
    timestamps = [sample.timestamp_ns for sample in stream]
    insertion = bisect.bisect_left(timestamps, timestamp_ns, lo=start_index)
    candidates = []
    if insertion < len(stream):
        candidates.append(insertion)
    if insertion - 1 >= start_index:
        candidates.append(insertion - 1)
    if not candidates:
        raise LookupError("no candidate")
    return min(
        candidates,
        key=lambda index: (
            abs(stream[index].timestamp_ns - timestamp_ns),
            stream[index].timestamp_ns,
        ),
    )


def nearest_sample(
    stream: list[TimestampedSource],
    timestamp_ns: int,
    max_delta_ns: int | None,
) -> TimestampedSource | None:
    try:
        sample = stream[nearest_index(stream, timestamp_ns)]
    except LookupError:
        return None
    if max_delta_ns is not None and abs(sample.timestamp_ns - timestamp_ns) > max_delta_ns:
        return None
    return sample


def previous_sample(
    stream: list[TimestampedSource],
    timestamp_ns: int,
    max_age_ns: int | None,
) -> TimestampedSource | None:
    timestamps = [sample.timestamp_ns for sample in stream]
    index = bisect.bisect_right(timestamps, timestamp_ns) - 1
    if index < 0:
        return None
    sample = stream[index]
    age = timestamp_ns - sample.timestamp_ns
    if max_age_ns is not None and age > max_age_ns:
        return None
    return sample


def build_alignment(
    rows: dict[str, list[Any]],
    max_depth_delta_ns: int | None,
    max_fisheye_delta_ns: int | None,
    max_state_delta_ns: int | None,
    max_action_age_ns: int | None,
) -> tuple[list[AlignedFrame], dict[str, int]]:
    color_stream = make_stream(rows, "observation_color")
    depth_stream = make_stream(rows, "observation_depth")
    fisheye_stream = make_stream(rows, "observation_fisheye")
    joint_stream = make_stream(rows, "observation_joint_state")
    gripper_stream = make_stream(rows, "observation_gripper")
    action_joint_stream = make_stream(rows, "action_joint_state")
    action_gripper_stream = make_stream(rows, "action_gripper")

    aligned: list[AlignedFrame] = []
    depth_cursor = 0
    fisheye_cursor = 0
    rejected = {
        "depth": 0,
        "fisheye": 0,
        "state": 0,
        "action": 0,
    }

    for color in color_stream:
        try:
            depth_index = nearest_index(depth_stream, color.timestamp_ns, depth_cursor)
        except LookupError:
            rejected["depth"] += 1
            continue
        depth = depth_stream[depth_index]
        if (
            max_depth_delta_ns is not None
            and abs(depth.timestamp_ns - color.timestamp_ns) > max_depth_delta_ns
        ):
            rejected["depth"] += 1
            continue

        try:
            fisheye_index = nearest_index(fisheye_stream, color.timestamp_ns, fisheye_cursor)
        except LookupError:
            rejected["fisheye"] += 1
            continue
        fisheye = fisheye_stream[fisheye_index]
        if (
            max_fisheye_delta_ns is not None
            and abs(fisheye.timestamp_ns - color.timestamp_ns) > max_fisheye_delta_ns
        ):
            rejected["fisheye"] += 1
            continue

        observation_joint = nearest_sample(joint_stream, color.timestamp_ns, max_state_delta_ns)
        observation_gripper = nearest_sample(gripper_stream, color.timestamp_ns, max_state_delta_ns)
        if observation_joint is None or observation_gripper is None:
            rejected["state"] += 1
            continue

        action_joint = previous_sample(action_joint_stream, color.timestamp_ns, max_action_age_ns)
        action_gripper = previous_sample(action_gripper_stream, color.timestamp_ns, max_action_age_ns)
        if action_joint is None or action_gripper is None:
            rejected["action"] += 1
            continue

        # Only commit image cursors when the complete aligned frame is accepted.
        depth_cursor = depth_index + 1
        fisheye_cursor = fisheye_index + 1
        aligned.append(
            AlignedFrame(
                color=color,
                depth=depth,
                fisheye=fisheye,
                observation_joint=observation_joint,
                observation_gripper=observation_gripper,
                action_joint=action_joint,
                action_gripper=action_gripper,
            )
        )

    if len({frame.depth.source_frame_index for frame in aligned}) != len(aligned):
        raise AssertionError("Depth 图像被重复分配")
    if len({frame.fisheye.source_frame_index for frame in aligned}) != len(aligned):
        raise AssertionError("Fisheye 图像被重复分配")
    if any(frame.action_joint.timestamp_ns > frame.color.timestamp_ns for frame in aligned):
        raise AssertionError("关节 action 匹配到了 RGB 未来的数据")
    if any(frame.action_gripper.timestamp_ns > frame.color.timestamp_ns for frame in aligned):
        raise AssertionError("夹爪 action 匹配到了 RGB 未来的数据")

    stats = {
        "recorded_rows": len(rows["frame_index"]),
        "unique_rgb_frames": len(color_stream),
        "matched_frames": len(aligned),
        **{f"rejected_{key}": value for key, value in rejected.items()},
    }
    return aligned, stats


def percentile_summary(values_ns: list[int]) -> dict[str, float]:
    if not values_ns:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    values_ms = np.asarray(values_ns, dtype=np.float64) / 1_000_000
    return {
        "p50_ms": round(float(np.percentile(values_ms, 50)), 4),
        "p95_ms": round(float(np.percentile(values_ms, 95)), 4),
        "max_ms": round(float(np.max(values_ms)), 4),
    }


def alignment_report(
    source_episode_index: int,
    output_episode_index: int | None,
    aligned: list[AlignedFrame],
    stats: dict[str, int],
) -> dict[str, Any]:
    return {
        "source_episode_index": source_episode_index,
        "output_episode_index": output_episode_index,
        **stats,
        "depth_abs_delta": percentile_summary(
            [abs(frame.depth.timestamp_ns - frame.color.timestamp_ns) for frame in aligned]
        ),
        "fisheye_abs_delta": percentile_summary(
            [abs(frame.fisheye.timestamp_ns - frame.color.timestamp_ns) for frame in aligned]
        ),
        "observation_joint_abs_delta": percentile_summary(
            [abs(frame.observation_joint.timestamp_ns - frame.color.timestamp_ns) for frame in aligned]
        ),
        "observation_gripper_abs_delta": percentile_summary(
            [abs(frame.observation_gripper.timestamp_ns - frame.color.timestamp_ns) for frame in aligned]
        ),
        "action_joint_age": percentile_summary(
            [frame.color.timestamp_ns - frame.action_joint.timestamp_ns for frame in aligned]
        ),
        "action_gripper_age": percentile_summary(
            [frame.color.timestamp_ns - frame.action_gripper.timestamp_ns for frame in aligned]
        ),
    }


def get_video_path(
    root: Path,
    info: dict[str, Any],
    episode_meta: dict[str, Any],
    video_key: str,
) -> Path:
    return root / info["video_path"].format(
        video_key=video_key,
        chunk_index=int(episode_meta[f"videos/{video_key}/chunk_index"]),
        file_index=int(episode_meta[f"videos/{video_key}/file_index"]),
    )


def video_file_frame_index(
    episode_meta: dict[str, Any],
    video_key: str,
    source_frame_index: int,
    fps: int,
) -> int:
    episode_start = float(episode_meta[f"videos/{video_key}/from_timestamp"])
    return int(round(episode_start * fps)) + source_frame_index


def numpy_feature_value(value: Any, feature: dict[str, Any]) -> np.ndarray:
    return np.asarray(value, dtype=np.dtype(feature["dtype"]))


def make_output_frame(
    source_data: dict[str, list[Any]],
    features: dict[str, dict[str, Any]],
    tasks: dict[int, str],
    aligned: AlignedFrame,
    color_image: np.ndarray,
    depth_image: np.ndarray,
    fisheye_image: np.ndarray,
) -> dict[str, Any]:
    color_row = aligned.color.source_frame_index
    joint_state = numpy_feature_value(
        source_data[STATE_KEY][aligned.observation_joint.source_frame_index], features[STATE_KEY]
    )
    gripper_state = numpy_feature_value(
        source_data[STATE_KEY][aligned.observation_gripper.source_frame_index], features[STATE_KEY]
    )
    joint_action = numpy_feature_value(
        source_data[ACTION_KEY][aligned.action_joint.source_frame_index], features[ACTION_KEY]
    )
    gripper_action = numpy_feature_value(
        source_data[ACTION_KEY][aligned.action_gripper.source_frame_index], features[ACTION_KEY]
    )
    state = joint_state.copy()
    state[-1] = gripper_state[-1]
    action = joint_action.copy()
    action[-1] = gripper_action[-1]

    frame: dict[str, Any] = {
        STATE_KEY: state,
        ACTION_KEY: action,
        RGB_KEY: color_image,
        DEPTH_KEY: depth_image,
        FISHEYE_KEY: fisheye_image,
        "task": tasks[int(source_data["task_index"][color_row])],
    }
    known_features = {STATE_KEY, ACTION_KEY, RGB_KEY, DEPTH_KEY, FISHEYE_KEY}
    for key, feature in features.items():
        if key in known_features or key in DEFAULT_FEATURES:
            continue
        if feature["dtype"] in {"video", "image"}:
            raise RuntimeError(f"暂不支持额外的图像特征：{key}")
        frame[key] = numpy_feature_value(source_data[key][color_row], feature)
    return frame


def make_ros_row(
    output_episode_index: int,
    output_frame_index: int,
    fps: int,
    frame: AlignedFrame,
) -> dict[str, int | float]:
    return {
        "episode_index": output_episode_index,
        "frame_index": output_frame_index,
        "lerobot_timestamp": output_frame_index / fps,
        "observation_joint_state": frame.observation_joint.timestamp_ns,
        "observation_gripper": frame.observation_gripper.timestamp_ns,
        "observation_color": frame.color.timestamp_ns,
        "observation_depth": frame.depth.timestamp_ns,
        "observation_fisheye": frame.fisheye.timestamp_ns,
        "action_joint_state": frame.action_joint.timestamp_ns,
        "action_gripper": frame.action_gripper.timestamp_ns,
    }


def save_ros_episode(root: Path, episode_index: int, rows: list[dict[str, int | float]]) -> None:
    directory = root / "ros_timestamps"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"episode_{episode_index:06d}.parquet"
    if path.exists():
        raise FileExistsError(f"拒绝覆盖 ROS 时间戳文件：{path}")
    temporary_path = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=ROS_SCHEMA), temporary_path)
    temporary_path.replace(path)


def validate_source(info: dict[str, Any]) -> None:
    features = info.get("features", {})
    required = {RGB_KEY, DEPTH_KEY, FISHEYE_KEY, STATE_KEY, ACTION_KEY}
    missing = required - set(features)
    if missing:
        raise RuntimeError(f"数据集缺少 Nero 对齐所需特征：{sorted(missing)}")
    if tuple(features[STATE_KEY]["shape"]) != (8,) or tuple(features[ACTION_KEY]["shape"]) != (8,):
        raise RuntimeError("Nero observation.state 和 action 必须均为 7 关节 + 1 夹爪")
    for key in (STATE_KEY, ACTION_KEY):
        names = features[key].get("names")
        if not names or len(names) != 8 or names[-1] != "gripper.pos":
            raise RuntimeError(f"{key} 的最后一维必须是 gripper.pos：{names}")
    video_keys = {key for key, feature in features.items() if feature["dtype"] == "video"}
    expected_video_keys = {RGB_KEY, DEPTH_KEY, FISHEYE_KEY}
    if video_keys != expected_video_keys:
        raise RuntimeError(
            f"视频特征必须恰好为 {sorted(expected_video_keys)}，当前为 {sorted(video_keys)}"
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"原数据集不存在：{source_root}")
    if not args.dry_run and args.output_root is None:
        raise ValueError("非 --dry-run 模式必须提供 --output-root")

    info = load_json(source_root / "meta" / "info.json")
    validate_source(info)
    fps = int(info["fps"])
    episode_metadata = load_episode_metadata(source_root)
    tasks = load_tasks(source_root)
    timestamp_store = RosTimestampStore(source_root)
    selected_episodes = select_episode_indices(
        args.episode,
        args.episodes,
        len(episode_metadata),
    )

    output_root = args.output_root.resolve() if args.output_root is not None else None
    destination = None
    if not args.dry_run:
        assert output_root is not None
        if output_root.exists():
            raise FileExistsError(f"为了保护数据，输出目录不能已存在：{output_root}")
        repo_id = args.output_repo_id or f"local/{output_root.name}"
        destination = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=normalize_features(info["features"]),
            root=output_root,
            robot_type=info.get("robot_type"),
            use_videos=True,
            vcodec=args.vcodec,
            streaming_encoding=True,
            encoder_threads=args.encoder_threads,
        )

    reports: list[dict[str, Any]] = []
    try:
        for source_episode_index in selected_episodes:
            episode_meta = episode_metadata[source_episode_index]
            ros_rows = timestamp_store.load_episode(source_episode_index)
            source_data = load_episode_data(source_root, info, episode_meta)
            expected_length = int(episode_meta["length"])
            if len(ros_rows["frame_index"]) != expected_length:
                raise RuntimeError(
                    f"episode {source_episode_index}: ROS 时间戳行数 "
                    f"{len(ros_rows['frame_index'])} != 数据帧数 {expected_length}"
                )
            if len(source_data["frame_index"]) != expected_length:
                raise RuntimeError(
                    f"episode {source_episode_index}: data 行数 "
                    f"{len(source_data['frame_index'])} != metadata length {expected_length}"
                )
            expected_frame_indices = list(range(expected_length))
            if [int(value) for value in ros_rows["frame_index"]] != expected_frame_indices:
                raise RuntimeError(
                    f"episode {source_episode_index}: ROS frame_index 不是从 0 连续递增"
                )
            if [int(value) for value in source_data["frame_index"]] != expected_frame_indices:
                raise RuntimeError(
                    f"episode {source_episode_index}: data frame_index 不是从 0 连续递增"
                )

            aligned, stats = build_alignment(
                ros_rows,
                milliseconds_to_ns(args.max_depth_delta_ms),
                milliseconds_to_ns(args.max_fisheye_delta_ms),
                milliseconds_to_ns(args.max_state_delta_ms),
                milliseconds_to_ns(args.max_action_age_ms),
            )
            if args.max_frames_per_episode > 0:
                aligned = aligned[: args.max_frames_per_episode]
                stats["matched_frames"] = len(aligned)
            output_episode_index = destination.meta.total_episodes if destination is not None else None
            report = alignment_report(
                source_episode_index,
                output_episode_index,
                aligned,
                stats,
            )
            reports.append(report)
            logging.info(
                "episode %d: recorded=%d, unique_rgb=%d, matched=%d",
                source_episode_index,
                stats["recorded_rows"],
                stats["unique_rgb_frames"],
                len(aligned),
            )
            if args.dry_run:
                continue
            if not aligned:
                logging.warning("episode %d 没有可用匹配，已跳过", source_episode_index)
                continue

            assert destination is not None and output_root is not None
            output_episode_index = destination.meta.total_episodes
            video_readers = {
                key: SequentialVideoReader(get_video_path(source_root, info, episode_meta, key))
                for key in (RGB_KEY, DEPTH_KEY, FISHEYE_KEY)
            }
            timestamp_output_rows = []
            try:
                for output_frame_index, match in enumerate(aligned):
                    color_image = video_readers[RGB_KEY].read(
                        video_file_frame_index(
                            episode_meta, RGB_KEY, match.color.source_frame_index, fps
                        )
                    )
                    depth_image = video_readers[DEPTH_KEY].read(
                        video_file_frame_index(
                            episode_meta, DEPTH_KEY, match.depth.source_frame_index, fps
                        )
                    )
                    fisheye_image = video_readers[FISHEYE_KEY].read(
                        video_file_frame_index(
                            episode_meta, FISHEYE_KEY, match.fisheye.source_frame_index, fps
                        )
                    )
                    destination.add_frame(
                        make_output_frame(
                            source_data,
                            info["features"],
                            tasks,
                            match,
                            color_image,
                            depth_image,
                            fisheye_image,
                        )
                    )
                    timestamp_output_rows.append(
                        make_ros_row(
                            output_episode_index,
                            output_frame_index,
                            fps,
                            match,
                        )
                    )
            finally:
                for reader in video_readers.values():
                    reader.close()

            destination.save_episode(parallel_encoding=False)
            save_ros_episode(output_root, output_episode_index, timestamp_output_rows)

        summary = {
            "source_root": str(source_root),
            "output_root": str(output_root) if output_root is not None else None,
            "rules": {
                "reference": "observation_color",
                "depth_and_fisheye": "monotonic nearest, one-to-one",
                "observation_state": "nearest, reuse allowed",
                "action": "latest timestamp <= RGB, reuse allowed",
                "max_depth_delta_ms": args.max_depth_delta_ms,
                "max_fisheye_delta_ms": args.max_fisheye_delta_ms,
                "max_state_delta_ms": args.max_state_delta_ms,
                "max_action_age_ms": args.max_action_age_ms,
            },
            "episodes": reports,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if output_root is not None:
            (output_root / "alignment_report.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    finally:
        if destination is not None:
            destination.finalize()


if __name__ == "__main__":
    main()
