#!/usr/bin/env python

"""Convert an oversampled Nero dataset to a requested RGB-based output rate.

The source LeRobot loop may run at 60 Hz, 80 Hz, or another rate. Repeated RGB
ROS timestamps are collapsed to one frame, while the extra samples remain
available when matching observation state and causal actions. The output
dataset is encoded at ``--output-fps``.
"""

from __future__ import annotations

import argparse
import json
import logging
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
    video_file_frame_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将任意高频采集的 Nero ROS 2 数据集，按 RGB ROS 时间戳去重并输出为指定频率。"
        )
    )
    parser.add_argument("--source-root", type=Path, required=True, help="原始 LeRobot 数据集目录")
    parser.add_argument("--output-root", type=Path, help="新数据集目录（不允许已存在）")
    parser.add_argument("--output-repo-id", help="新数据集 repo_id")
    parser.add_argument(
        "--output-fps",
        type=int,
        required=True,
        help="最终 LeRobot 数据集和三路视频的频率，例如 30",
    )
    episode_group = parser.add_mutually_exclusive_group()
    episode_group.add_argument(
        "--episode",
        default="all",
        metavar="INDEX|all",
        help="处理指定 episode_index，或使用 all 处理全部（默认）",
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
        help="反馈状态与 RGB 最大绝对时间差；<=0 不限制",
    )
    parser.add_argument(
        "--max-action-age-ms",
        type=float,
        default=0.0,
        help="Action 允许比 RGB 旧的最大时间；<=0 不限制",
    )
    parser.add_argument("--vcodec", default="libsvtav1", help="输出视频编码器")
    parser.add_argument("--encoder-threads", type=int, help="每路视频编码线程数")
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=0,
        help="仅输出每轮前 N 个匹配，用于验证；0 表示全部",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只计算匹配报告，不解码视频、不写数据集",
    )
    return parser.parse_args()


def observed_rgb_fps(ros_rows: dict[str, list[Any]]) -> float:
    color_stream = make_stream(ros_rows, "observation_color")
    if len(color_stream) < 2:
        return 0.0
    duration_s = (color_stream[-1].timestamp_ns - color_stream[0].timestamp_ns) / 1_000_000_000
    return 0.0 if duration_s <= 0 else (len(color_stream) - 1) / duration_s


def make_resampled_features(info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Remove source video metadata so LeRobot probes the newly encoded rate."""
    features = normalize_features(info["features"])
    for feature in features.values():
        if feature["dtype"] == "video":
            feature.pop("info", None)
    return features


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"原数据集不存在：{source_root}")
    if args.output_fps <= 0:
        raise ValueError(f"--output-fps 必须为正整数：{args.output_fps}")
    if not args.dry_run and args.output_root is None:
        raise ValueError("非 --dry-run 模式必须提供 --output-root")

    info = load_json(source_root / "meta" / "info.json")
    validate_source(info)
    source_fps = int(info["fps"])
    output_fps = int(args.output_fps)
    if source_fps <= 0:
        raise RuntimeError(f"原数据集 fps 无效：{source_fps}")
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
        destination = LeRobotDataset.create(
            repo_id=args.output_repo_id or f"local/{output_root.name}",
            fps=output_fps,
            features=make_resampled_features(info),
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
            expected_indices = list(range(expected_length))
            if [int(value) for value in ros_rows["frame_index"]] != expected_indices:
                raise RuntimeError(
                    f"episode {source_episode_index}: ROS frame_index 与 metadata length 不一致"
                )
            if [int(value) for value in source_data["frame_index"]] != expected_indices:
                raise RuntimeError(
                    f"episode {source_episode_index}: data frame_index 与 metadata length 不一致"
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
            rgb_fps = observed_rgb_fps(ros_rows)
            report["observed_unique_rgb_fps"] = round(rgb_fps, 4)
            reports.append(report)
            logging.info(
                "episode %d: source=%d Hz, unique RGB=%.3f Hz, output=%d Hz, matched=%d",
                source_episode_index,
                source_fps,
                rgb_fps,
                output_fps,
                len(aligned),
            )
            if rgb_fps > 0 and abs(rgb_fps - output_fps) / output_fps > 0.1:
                logging.warning(
                    "episode %d 去重后 RGB 约 %.3f Hz，与输出 %d Hz 相差超过 10%%；"
                    "输出会按 %d Hz 播放，请确认目标频率",
                    source_episode_index,
                    rgb_fps,
                    output_fps,
                    output_fps,
                )
            if args.dry_run:
                continue
            if not aligned:
                logging.warning("episode %d 没有可用匹配，已跳过", source_episode_index)
                continue

            assert destination is not None and output_root is not None
            output_episode_index = destination.meta.total_episodes
            readers = {
                key: SequentialVideoReader(get_video_path(source_root, info, episode_meta, key))
                for key in (RGB_KEY, DEPTH_KEY, FISHEYE_KEY)
            }
            output_ros_rows: list[dict[str, int | float]] = []
            try:
                for output_frame_index, match in enumerate(aligned):
                    color_image = readers[RGB_KEY].read(
                        video_file_frame_index(
                            episode_meta,
                            RGB_KEY,
                            match.color.source_frame_index,
                            source_fps,
                        )
                    )
                    depth_image = readers[DEPTH_KEY].read(
                        video_file_frame_index(
                            episode_meta,
                            DEPTH_KEY,
                            match.depth.source_frame_index,
                            source_fps,
                        )
                    )
                    fisheye_image = readers[FISHEYE_KEY].read(
                        video_file_frame_index(
                            episode_meta,
                            FISHEYE_KEY,
                            match.fisheye.source_frame_index,
                            source_fps,
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
                    output_ros_rows.append(
                        make_ros_row(
                            output_episode_index,
                            output_frame_index,
                            output_fps,
                            match,
                        )
                    )
            finally:
                for reader in readers.values():
                    reader.close()

            destination.save_episode(parallel_encoding=False)
            save_ros_episode(output_root, output_episode_index, output_ros_rows)

        summary = {
            "source_root": str(source_root),
            "output_root": str(output_root) if output_root is not None else None,
            "source_recording_fps": source_fps,
            "output_dataset_fps": output_fps,
            "rules": {
                "reference": "deduplicated observation_color ROS timestamp",
                "depth_and_fisheye": "monotonic nearest, one-to-one",
                "observation_state": "nearest, reuse allowed",
                "action": "latest timestamp <= RGB, reuse allowed; future action forbidden",
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
