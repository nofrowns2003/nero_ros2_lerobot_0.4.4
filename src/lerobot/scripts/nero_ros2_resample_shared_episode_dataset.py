#!/usr/bin/env python3

"""Synchronize Nero episodes stored together in shared Parquet/video files.

The recording loop may append many episodes to the same LeRobot v3 Parquet
and MP4 files.  This script still aligns every episode independently using its
own ROS timestamp file, while resolving image indices inside the shared MP4
from the per-episode ``from_timestamp`` metadata.

Matching rules:

* deduplicated RGB timestamps are the reference stream;
* Depth and fisheye use monotonic nearest-neighbour one-to-one matching;
* observation joint/gripper state use nearest-neighbour matching and may reuse;
* action joint/gripper use causal zero-order hold and can never come from the
  future relative to RGB;
* the matched RGB, Depth and fisheye images are decoded and physically written
  into the output dataset.  This is not a timestamp-only rewrite.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.nero_ros2_align_dataset import (
    DEPTH_KEY,
    FISHEYE_KEY,
    RGB_KEY,
    RosTimestampStore,
    SequentialVideoReader,
    alignment_report,
    build_alignment,
    get_video_path,
    load_episode_data,
    load_episode_metadata,
    load_json,
    load_tasks,
    make_output_frame,
    make_ros_row,
    make_stream,
    milliseconds_to_ns,
    normalize_features,
    save_ros_episode,
    select_episode_indices,
    validate_source,
)


VIDEO_KEYS = (RGB_KEY, DEPTH_KEY, FISHEYE_KEY)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "同步 Nero ROS 2 多 episode 共享 Parquet/MP4 的 LeRobot v3 数据集，"
            "并按指定 FPS 输出。"
        )
    )
    parser.add_argument("--source-root", type=Path, required=True, help="原始数据集目录")
    parser.add_argument("--output-root", type=Path, help="新数据集目录（不允许已存在）")
    parser.add_argument("--output-repo-id", help="输出数据集 repo_id")
    parser.add_argument(
        "--output-fps",
        type=int,
        required=True,
        help="输出数据集及三路视频 FPS，例如 30",
    )
    episode_group = parser.add_mutually_exclusive_group()
    episode_group.add_argument(
        "--episode",
        default="all",
        metavar="INDEX|all",
        help="处理指定源 episode_index；all 处理全部（默认）",
    )
    episode_group.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-depth-delta-ms",
        type=float,
        default=20.0,
        help="Depth 与 RGB 最大绝对时间差；<=0 不限制",
    )
    parser.add_argument(
        "--max-fisheye-delta-ms",
        type=float,
        default=20.0,
        help="Fisheye 与 RGB 最大绝对时间差；<=0 不限制",
    )
    parser.add_argument(
        "--max-state-delta-ms",
        type=float,
        default=50.0,
        help="反馈关节/夹爪与 RGB 最大绝对时间差；<=0 不限制",
    )
    parser.add_argument(
        "--max-action-age-ms",
        type=float,
        default=0.0,
        help="Action 允许早于 RGB 的最大时间；<=0 不限制，未来始终禁止",
    )
    parser.add_argument("--vcodec", default="libsvtav1", help="输出视频编码器")
    parser.add_argument("--encoder-threads", type=int, help="每路视频编码线程数")
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=0,
        help="每轮最多输出 N 帧，用于小规模验证；0 表示全部",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只计算逐轮匹配和报告，不解码图像、不写输出数据集",
    )
    return parser.parse_args()


class SharedVideoReaderPool:
    """Reuse one sequential decoder for episodes sharing the same MP4 file."""

    def __init__(self) -> None:
        self._readers: dict[Path, SequentialVideoReader] = {}

    def read(self, path: Path, frame_index: int):
        reader = self._readers.get(path)
        if reader is None:
            logging.info("打开共享视频：%s", path)
            reader = SequentialVideoReader(path)
            self._readers[path] = reader
        return reader.read(frame_index)

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self) -> SharedVideoReaderPool:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def observed_rgb_fps(ros_rows: dict[str, list[Any]]) -> float:
    color_stream = make_stream(ros_rows, "observation_color")
    if len(color_stream) < 2:
        return 0.0
    duration_s = (color_stream[-1].timestamp_ns - color_stream[0].timestamp_ns) / 1_000_000_000
    return 0.0 if duration_s <= 0 else (len(color_stream) - 1) / duration_s


def make_output_features(info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Let LeRobot probe fresh video metadata at the requested output FPS."""
    features = normalize_features(info["features"])
    for feature in features.values():
        if feature["dtype"] == "video":
            feature.pop("info", None)
    return features


def validate_episode_rows(
    episode_index: int,
    expected_length: int,
    ros_rows: dict[str, list[Any]],
    source_data: dict[str, list[Any]],
) -> None:
    expected = list(range(expected_length))
    ros_indices = [int(value) for value in ros_rows["frame_index"]]
    data_indices = [int(value) for value in source_data["frame_index"]]
    if ros_indices != expected:
        raise RuntimeError(
            f"episode {episode_index}: ROS frame_index 必须为 0..{expected_length - 1}，"
            f"实际共 {len(ros_indices)} 行"
        )
    if data_indices != expected:
        raise RuntimeError(
            f"episode {episode_index}: data frame_index 必须为 0..{expected_length - 1}，"
            f"实际共 {len(data_indices)} 行"
        )


def shared_video_frame_index(
    episode_meta: dict[str, Any],
    video_key: str,
    local_frame_index: int,
    source_fps: int,
) -> int:
    """Map an episode-local image index to its frame in a shared MP4."""
    episode_length = int(episode_meta["length"])
    if local_frame_index < 0 or local_frame_index >= episode_length:
        raise IndexError(
            f"episode {episode_meta['episode_index']} {video_key}: 局部帧索引越界 "
            f"{local_frame_index}，episode length={episode_length}"
        )
    start_s = float(episode_meta[f"videos/{video_key}/from_timestamp"])
    end_s = float(episode_meta[f"videos/{video_key}/to_timestamp"])
    start_frame = int(round(start_s * source_fps))
    end_frame = int(round(end_s * source_fps))
    target = start_frame + local_frame_index
    if target >= end_frame:
        raise RuntimeError(
            f"episode {episode_meta['episode_index']} {video_key}: metadata 视频区间 "
            f"[{start_frame}, {end_frame}) 无法容纳局部帧 {local_frame_index}"
        )
    return target


def describe_shared_files(
    source_root: Path,
    info: dict[str, Any],
    episode_metadata: list[dict[str, Any]],
    selected_episodes: list[int],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    selected_set = set(selected_episodes)
    for video_key in VIDEO_KEYS:
        groups: dict[str, list[int]] = defaultdict(list)
        for episode_index in range(len(episode_metadata)):
            path = get_video_path(source_root, info, episode_metadata[episode_index], video_key)
            groups[str(path.relative_to(source_root))].append(episode_index)
        result[video_key] = [
            {
                "path": path,
                "source_episodes": episodes,
                "selected_source_episodes": [
                    episode for episode in episodes if episode in selected_set
                ],
                "shared": len(episodes) > 1,
            }
            for path, episodes in groups.items()
            if any(episode in selected_set for episode in episodes)
        ]
    return result


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    source_root = args.source_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"原数据集不存在：{source_root}")
    if args.output_fps <= 0:
        raise ValueError(f"--output-fps 必须为正整数：{args.output_fps}")
    if args.max_frames_per_episode < 0:
        raise ValueError("--max-frames-per-episode 不能为负数")
    if not args.dry_run and args.output_root is None:
        raise ValueError("非 --dry-run 模式必须提供 --output-root")

    info = load_json(source_root / "meta" / "info.json")
    validate_source(info)
    source_fps = int(info["fps"])
    output_fps = int(args.output_fps)
    episode_metadata = load_episode_metadata(source_root)
    tasks = load_tasks(source_root)
    timestamps = RosTimestampStore(source_root)
    selected_episodes = sorted(
        select_episode_indices(args.episode, args.episodes, len(episode_metadata))
    )
    shared_files = describe_shared_files(
        source_root, info, episode_metadata, selected_episodes
    )

    output_root = args.output_root.expanduser().resolve() if args.output_root else None
    destination = None
    if not args.dry_run:
        assert output_root is not None
        if output_root.exists():
            raise FileExistsError(f"为了保护数据，输出目录不能已存在：{output_root}")
        destination = LeRobotDataset.create(
            repo_id=args.output_repo_id or f"local/{output_root.name}",
            fps=output_fps,
            features=make_output_features(info),
            root=output_root,
            robot_type=info.get("robot_type"),
            use_videos=True,
            vcodec=args.vcodec,
            streaming_encoding=True,
            encoder_threads=args.encoder_threads,
        )

    reports: list[dict[str, Any]] = []
    try:
        with SharedVideoReaderPool() as video_pool:
            for source_episode_index in selected_episodes:
                episode_meta = episode_metadata[source_episode_index]
                ros_rows = timestamps.load_episode(source_episode_index)
                source_data = load_episode_data(source_root, info, episode_meta)
                expected_length = int(episode_meta["length"])
                validate_episode_rows(
                    source_episode_index,
                    expected_length,
                    ros_rows,
                    source_data,
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

                output_episode_index = (
                    destination.meta.total_episodes
                    if destination is not None and aligned
                    else None
                )
                report = alignment_report(
                    source_episode_index,
                    output_episode_index,
                    aligned,
                    stats,
                )
                report["observed_unique_rgb_fps"] = round(observed_rgb_fps(ros_rows), 4)
                reports.append(report)
                logging.info(
                    "episode %d: rows=%d, unique RGB=%d, matched=%d, output episode=%s",
                    source_episode_index,
                    expected_length,
                    stats["unique_rgb_frames"],
                    len(aligned),
                    output_episode_index,
                )

                if args.dry_run:
                    continue
                if not aligned:
                    logging.warning("episode %d 没有有效匹配，已跳过", source_episode_index)
                    continue

                assert destination is not None and output_root is not None
                assert output_episode_index is not None
                video_paths = {
                    key: get_video_path(source_root, info, episode_meta, key)
                    for key in VIDEO_KEYS
                }
                output_ros_rows: list[dict[str, int | float]] = []
                for output_frame_index, match in enumerate(aligned):
                    color_image = video_pool.read(
                        video_paths[RGB_KEY],
                        shared_video_frame_index(
                            episode_meta,
                            RGB_KEY,
                            match.color.source_frame_index,
                            source_fps,
                        ),
                    )
                    depth_image = video_pool.read(
                        video_paths[DEPTH_KEY],
                        shared_video_frame_index(
                            episode_meta,
                            DEPTH_KEY,
                            match.depth.source_frame_index,
                            source_fps,
                        ),
                    )
                    fisheye_image = video_pool.read(
                        video_paths[FISHEYE_KEY],
                        shared_video_frame_index(
                            episode_meta,
                            FISHEYE_KEY,
                            match.fisheye.source_frame_index,
                            source_fps,
                        ),
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
                    output_ros_rows.append(
                        make_ros_row(
                            output_episode_index,
                            output_frame_index,
                            output_fps,
                            match,
                        )
                    )
                    if (output_frame_index + 1) % 100 == 0:
                        logging.info(
                            "episode %d: 已重写图像和数据 %d/%d 帧",
                            source_episode_index,
                            output_frame_index + 1,
                            len(aligned),
                        )

                destination.save_episode(parallel_encoding=False)
                save_ros_episode(output_root, output_episode_index, output_ros_rows)
                logging.info(
                    "episode %d -> %d 保存完成，共 %d 帧",
                    source_episode_index,
                    output_episode_index,
                    len(aligned),
                )

        summary = {
            "source_root": str(source_root),
            "output_root": str(output_root) if output_root is not None else None,
            "source_recording_fps": source_fps,
            "output_dataset_fps": output_fps,
            "selected_source_episodes": selected_episodes,
            "shared_source_video_files": shared_files,
            "rules": {
                "reference": "deduplicated observation_color ROS timestamp",
                "depth_and_fisheye": "monotonic nearest, one-to-one; image reuse forbidden",
                "observation_state": "nearest, reuse allowed",
                "action": "latest timestamp <= RGB, reuse allowed; future action forbidden",
                "images": "RGB/Depth/Fisheye source frames are decoded and rewritten",
                "max_depth_delta_ms": args.max_depth_delta_ms,
                "max_fisheye_delta_ms": args.max_fisheye_delta_ms,
                "max_state_delta_ms": args.max_state_delta_ms,
                "max_action_age_ms": args.max_action_age_ms,
            },
            "episodes": reports,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if output_root is not None:
            (output_root / "shared_episode_alignment_report.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    finally:
        if destination is not None:
            destination.finalize()


if __name__ == "__main__":
    main()
