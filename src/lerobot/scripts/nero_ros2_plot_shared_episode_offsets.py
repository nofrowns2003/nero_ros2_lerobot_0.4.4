#!/usr/bin/env python3

"""Plot signed ROS timestamp offsets for a multi-episode Nero dataset.

The timestamp input remains one Parquet per episode even when LeRobot stores
all official data/video episodes in shared files.  ``--episode all`` creates
one PNG for every episode; an integer creates only that episode's PNG.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

# Do not alter or depend on the user's Matplotlib cache directory.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nero_ros2_matplotlib")
)

import matplotlib.pyplot as plt

from lerobot.scripts.nero_ros2_plot_timestamp_offsets import (
    TimestampStore,
    plot_episode,
    select_episodes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "为多 episode Nero 数据集逐轮绘制各 ROS 话题相对 RGB 的有符号时间差。"
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="同步处理后的 LeRobot 数据集根目录",
    )
    parser.add_argument(
        "--episode",
        default="all",
        metavar="INDEX|all",
        help="指定 episode_index；all 表示每轮各画一张（默认）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="图片目录；默认 DATASET_ROOT/shared_episode_timestamp_plots",
    )
    parser.add_argument(
        "--x-major-s",
        type=float,
        default=1.0,
        help="横轴主刻度间隔，单位秒（默认 1）",
    )
    parser.add_argument(
        "--y-limit-ms",
        type=float,
        default=100.0,
        help="纵轴显示范围 ±N ms（默认 ±100）",
    )
    parser.add_argument(
        "--y-major-ms",
        type=float,
        default=20.0,
        help="纵轴主刻度间隔，单位 ms（默认 20）",
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG DPI（默认 180）")
    parser.add_argument(
        "--show",
        action="store_true",
        help="保存完图片后显示 Matplotlib 窗口",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在：{dataset_root}")
    if args.x_major_s <= 0 or args.y_limit_ms <= 0 or args.y_major_ms <= 0:
        raise ValueError("横纵轴刻度和纵轴范围必须为正数")
    if args.dpi <= 0:
        raise ValueError("--dpi 必须为正数")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset_root / "shared_episode_timestamp_plots"
    )
    store = TimestampStore(dataset_root)
    available = store.episode_indices()
    episodes = select_episodes(args.episode, available)

    reports = []
    saved_paths = []
    for episode_index in episodes:
        output_path, report = plot_episode(
            dataset_root=dataset_root,
            output_dir=output_dir,
            episode_index=episode_index,
            rows=store.load(episode_index),
            x_major_s=args.x_major_s,
            y_limit_ms=args.y_limit_ms,
            y_major_ms=args.y_major_ms,
            dpi=args.dpi,
        )
        reports.append(report)
        saved_paths.append(str(output_path))
        print(f"saved episode {episode_index}: {output_path}")
        if not args.show:
            plt.close("all")

    summary = {
        "dataset_root": str(dataset_root),
        "requested_episode": args.episode,
        "available_episodes": available,
        "plotted_episodes": episodes,
        "offset_definition": "topic_timestamp - observation_color_timestamp",
        "x_major_s": args.x_major_s,
        "y_limit_ms": args.y_limit_ms,
        "y_major_ms": args.y_major_ms,
        "figures": saved_paths,
        "episodes": reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "timestamp_offset_report.json"
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved report: {report_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
