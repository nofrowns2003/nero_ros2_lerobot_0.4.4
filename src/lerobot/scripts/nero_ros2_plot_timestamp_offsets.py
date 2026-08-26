#!/usr/bin/env python3
"""Plot signed ROS timestamp offsets relative to RGB for aligned Nero datasets.

For every output frame, the plotted value is::

    offset_ms = (topic_timestamp_ns - observation_color_timestamp_ns) / 1e6

A negative value means that the topic sample is earlier than RGB; a positive
value means that it is later than RGB. One figure is generated per episode.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

# Avoid writing Matplotlib's cache into the user's home directory.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nero_ros2_matplotlib")
)

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from matplotlib.ticker import MultipleLocator


RGB_COLUMN = "observation_color"
TOPICS: tuple[tuple[str, str, str], ...] = (
    ("observation_depth", "Observation depth", "tab:blue"),
    ("observation_fisheye", "Observation fisheye", "tab:orange"),
    ("observation_joint_state", "Observation joint state", "tab:green"),
    ("observation_gripper", "Observation gripper", "tab:red"),
    ("action_joint_state", "Action joint state", "tab:purple"),
    ("action_gripper", "Action gripper", "tab:brown"),
)
REQUIRED_COLUMNS = (
    "episode_index",
    "frame_index",
    RGB_COLUMN,
    *(column for column, _, _ in TOPICS),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取整理后的 Nero ROS 时间戳，以 RGB 时刻为横轴，绘制其他话题相对 RGB 的有符号时间差。"
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="整理后的 LeRobot 数据集根目录",
    )
    parser.add_argument(
        "--episode",
        default="all",
        metavar="INDEX|all",
        help="输出 episode_index，例如 0；all 表示全部（默认）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="图片输出目录；默认 DATASET_ROOT/timestamp_offset_plots",
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
        help="纵轴范围 ±N ms（默认 ±100）",
    )
    parser.add_argument(
        "--y-major-ms",
        type=float,
        default=20.0,
        help="纵轴主刻度间隔，单位 ms（默认 20）",
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG 分辨率（默认 180）")
    parser.add_argument(
        "--show",
        action="store_true",
        help="保存后打开 Matplotlib 窗口",
    )
    return parser.parse_args()


class TimestampStore:
    def __init__(self, dataset_root: Path):
        self.directory = dataset_root / "ros_timestamps"
        self.legacy_path = self.directory / "ros2_timestamp.parquet"
        self.episode_paths = sorted(self.directory.glob("episode_*.parquet"))
        if not self.episode_paths and not self.legacy_path.exists():
            raise FileNotFoundError(f"找不到 ROS 时间戳 Parquet：{self.directory}")

    def episode_indices(self) -> list[int]:
        if self.episode_paths:
            indices = []
            for path in self.episode_paths:
                try:
                    indices.append(int(path.stem.removeprefix("episode_")))
                except ValueError as error:
                    raise RuntimeError(f"无法解析 episode 文件名：{path.name}") from error
            return indices
        table = pq.read_table(self.legacy_path, columns=["episode_index"])
        return sorted({int(value) for value in table["episode_index"].to_pylist()})

    def load(self, episode_index: int) -> dict[str, list[Any]]:
        episode_path = self.directory / f"episode_{episode_index:06d}.parquet"
        if episode_path.exists():
            table = pq.read_table(episode_path)
        elif self.legacy_path.exists():
            table = pq.read_table(self.legacy_path)
            table = table.filter(pc.equal(table["episode_index"], episode_index))
        else:
            raise FileNotFoundError(f"找不到 episode {episode_index}：{episode_path}")

        missing = set(REQUIRED_COLUMNS) - set(table.column_names)
        if missing:
            raise RuntimeError(f"{episode_path} 缺少列：{sorted(missing)}")
        table = table.select(REQUIRED_COLUMNS).sort_by("frame_index")
        if table.num_rows == 0:
            raise RuntimeError(f"episode {episode_index} 没有数据")
        episode_values = {int(value) for value in table["episode_index"].to_pylist()}
        if episode_values != {episode_index}:
            raise RuntimeError(
                f"episode {episode_index} 文件包含其他 episode：{sorted(episode_values)}"
            )
        return table.to_pydict()


def select_episodes(requested: str, available: list[int]) -> list[int]:
    if requested.lower() == "all":
        return available
    try:
        selected = int(requested)
    except ValueError as error:
        raise ValueError(f"--episode 必须是非负整数或 all：{requested!r}") from error
    if selected not in available:
        raise ValueError(f"episode {selected} 不存在；可用 episode：{available}")
    return [selected]


def signed_offset_summary(offset_ms: np.ndarray, y_limit_ms: float) -> dict[str, float | int]:
    return {
        "min_ms": round(float(np.min(offset_ms)), 4),
        "p50_ms": round(float(np.percentile(offset_ms, 50)), 4),
        "p95_ms": round(float(np.percentile(offset_ms, 95)), 4),
        "max_ms": round(float(np.max(offset_ms)), 4),
        "below_negative_limit": int(np.sum(offset_ms < -y_limit_ms)),
        "above_positive_limit": int(np.sum(offset_ms > y_limit_ms)),
    }


def plot_episode(
    dataset_root: Path,
    output_dir: Path,
    episode_index: int,
    rows: dict[str, list[Any]],
    x_major_s: float,
    y_limit_ms: float,
    y_major_ms: float,
    dpi: int,
) -> tuple[Path, dict[str, Any]]:
    rgb_ns = np.asarray(rows[RGB_COLUMN], dtype=np.int64)
    if np.any(rgb_ns <= 0):
        raise RuntimeError(f"episode {episode_index}: RGB 包含非正时间戳")
    if np.any(np.diff(rgb_ns) < 0):
        raise RuntimeError(f"episode {episode_index}: RGB 时间戳发生倒退")

    rgb_time_s = (rgb_ns - rgb_ns[0]).astype(np.float64) / 1_000_000_000.0
    figure, axes = plt.subplots(
        len(TOPICS),
        1,
        figsize=(16, 14),
        sharex=True,
        constrained_layout=True,
    )
    figure.suptitle(
        f"ROS timestamp offsets relative to RGB | episode {episode_index:06d}\n"
        "offset = topic timestamp - RGB timestamp",
        fontsize=14,
    )

    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "episode_index": episode_index,
        "frames": len(rgb_ns),
        "rgb_start_ns": int(rgb_ns[0]),
        "rgb_end_ns": int(rgb_ns[-1]),
        "duration_s": round(float(rgb_time_s[-1]), 6),
        "offset_definition": "topic_timestamp - observation_color_timestamp",
        "topics": {},
    }

    for axis, (column, label, color) in zip(axes, TOPICS, strict=True):
        topic_ns = np.asarray(rows[column], dtype=np.int64)
        if len(topic_ns) != len(rgb_ns):
            raise RuntimeError(
                f"episode {episode_index}: {column} 行数 {len(topic_ns)} != RGB 行数 {len(rgb_ns)}"
            )
        offset_ms = (topic_ns - rgb_ns).astype(np.float64) / 1_000_000.0
        summary = signed_offset_summary(offset_ms, y_limit_ms)
        report["topics"][column] = summary

        axis.plot(
            rgb_time_s,
            offset_ms,
            color=color,
            linewidth=0.9,
            marker=".",
            markersize=2.5,
        )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.75)
        axis.set_ylim(-y_limit_ms, y_limit_ms)
        axis.yaxis.set_major_locator(MultipleLocator(y_major_ms))
        axis.xaxis.set_major_locator(MultipleLocator(x_major_s))
        axis.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.55)
        outside = summary["below_negative_limit"] + summary["above_positive_limit"]
        outside_text = f", outside ±{y_limit_ms:g} ms: {outside}" if outside else ""
        axis.set_title(
            f"{label} | min={summary['min_ms']:.3f}, "
            f"p50={summary['p50_ms']:.3f}, max={summary['max_ms']:.3f} ms{outside_text}",
            fontsize=10,
        )
        axis.set_ylabel("Offset (ms)")

    axes[-1].set_xlabel("RGB time from first frame (s)")
    if rgb_time_s[-1] > 0:
        axes[-1].set_xlim(0.0, float(rgb_time_s[-1]))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"episode_{episode_index:06d}_timestamp_offsets.png"
    figure.savefig(output_path, dpi=dpi)
    return output_path, report


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在：{dataset_root}")
    if args.x_major_s <= 0 or args.y_limit_ms <= 0 or args.y_major_ms <= 0:
        raise ValueError("刻度间隔和纵轴范围必须为正数")
    if args.dpi <= 0:
        raise ValueError("--dpi 必须为正数")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset_root / "timestamp_offset_plots"
    )
    store = TimestampStore(dataset_root)
    episodes = select_episodes(args.episode, store.episode_indices())

    reports = []
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
        print(f"saved: {output_path}")
        print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
