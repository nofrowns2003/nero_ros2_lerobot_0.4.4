#!/usr/bin/env python3

"""Delete and reindex Nero ROS timestamp sidecar files after dataset editing.

This companion command intentionally mirrors the relevant arguments of
``lerobot-edit-dataset``.  It reads ``<root>/ros_timestamps`` and writes the
matching sidecars into ``<root>/<new_repo_id>/ros_timestamps``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


EPISODE_FILE_RE = re.compile(r"episode_(\d{6})\.parquet")


def parse_episode_indices(value: str) -> list[int]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"episode_indices 不是有效 JSON 列表：{exc}") from exc

    if not isinstance(parsed, list) or not parsed:
        raise argparse.ArgumentTypeError("episode_indices 必须是非空 JSON 列表，例如 '[25, 26]' ")
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in parsed):
        raise argparse.ArgumentTypeError("episode_indices 只能包含非负整数")
    if len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError("episode_indices 不能包含重复序号")
    return sorted(parsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为 delete_episodes 后的新数据集同步删除并重排 ros_timestamps。"
    )
    parser.add_argument("--repo_id", required=True, help="源数据集 repo_id，仅用于一致性检查和日志。")
    parser.add_argument("--root", type=Path, required=True, help="源数据集根目录。")
    parser.add_argument("--new_repo_id", required=True, help="与 lerobot-edit-dataset 相同的新 repo_id。")
    parser.add_argument(
        "--operation.type",
        dest="operation_type",
        required=True,
        choices=("delete_episodes",),
    )
    parser.add_argument(
        "--operation.episode_indices",
        dest="episode_indices",
        required=True,
        type=parse_episode_indices,
    )
    parser.add_argument("--dry-run", action="store_true", help="只检查并显示映射，不写文件。")
    return parser.parse_args()


def load_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"缺少数据集元数据：{info_path}")
    with info_path.open(encoding="utf-8") as file:
        return json.load(file)


def resolve_output_root(root: Path, new_repo_id: str) -> Path:
    repo_path = Path(new_repo_id)
    if repo_path.is_absolute() or ".." in repo_path.parts:
        raise ValueError("new_repo_id 必须是相对 repo id，且不能包含 '..'")
    return root / repo_path


def indexed_timestamp_files(directory: Path) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"时间戳目录不存在：{directory}")

    indexed: dict[int, Path] = {}
    unexpected: list[str] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = EPISODE_FILE_RE.fullmatch(path.name)
        if match is None:
            unexpected.append(path.name)
            continue
        index = int(match.group(1))
        if index in indexed:
            raise ValueError(f"episode {index} 存在重复时间戳文件")
        indexed[index] = path

    if unexpected:
        raise ValueError(f"时间戳目录包含无法识别的文件：{sorted(unexpected)}")
    return indexed


def validate_internal_episode_index(path: Path, expected_index: int) -> pq.ParquetFile:
    parquet_file = pq.ParquetFile(path)
    if "episode_index" not in parquet_file.schema_arrow.names:
        raise ValueError(f"{path.name} 缺少内部 episode_index 列")

    indices = pq.read_table(path, columns=["episode_index"])["episode_index"]
    unique_indices = set(indices.to_pylist())
    if unique_indices != {expected_index}:
        raise ValueError(
            f"{path.name} 内部 episode_index 应为 {expected_index}，实际为 {sorted(unique_indices)}"
        )
    return parquet_file


def rewrite_episode(path: Path, output_path: Path, old_index: int, new_index: int) -> None:
    validate_internal_episode_index(path, old_index)
    table = pq.read_table(path)
    column_position = table.schema.get_field_index("episode_index")
    field = table.schema.field(column_position)
    replacement = pa.array([new_index] * table.num_rows, type=field.type)
    table = table.set_column(column_position, field, replacement)
    pq.write_table(table, output_path)


def synchronize_timestamps(args: argparse.Namespace) -> None:
    del args.operation_type  # argparse choices already guarantees delete_episodes.
    source_root = args.root.expanduser().resolve()
    output_root = resolve_output_root(source_root, args.new_repo_id)
    source_info = load_info(source_root)
    output_info = load_info(output_root)

    source_repo_id = source_info.get("repo_id")
    if source_repo_id is not None and source_repo_id != args.repo_id:
        raise ValueError(f"源数据集 repo_id={source_repo_id!r}，与参数 {args.repo_id!r} 不一致")

    source_total = int(source_info["total_episodes"])
    output_total = int(output_info["total_episodes"])
    source_dir = source_root / "ros_timestamps"
    output_dir = output_root / "ros_timestamps"
    indexed_files = indexed_timestamp_files(source_dir)

    expected_source_indices = set(range(source_total))
    actual_source_indices = set(indexed_files)
    if actual_source_indices != expected_source_indices:
        missing = sorted(expected_source_indices - actual_source_indices)
        extra = sorted(actual_source_indices - expected_source_indices)
        raise ValueError(f"源时间戳序号不连续；缺少={missing}，多出={extra}")

    deleted = set(args.episode_indices)
    invalid = sorted(deleted - expected_source_indices)
    if invalid:
        raise ValueError(f"待删除 episode 超出源数据集范围：{invalid}")
    kept = [index for index in range(source_total) if index not in deleted]
    if len(kept) != output_total:
        raise ValueError(
            "删除后的时间戳数量与新数据集不一致："
            f"{source_total} - {len(deleted)} = {len(kept)}，"
            f"但新数据集声明 {output_total} episodes"
        )

    mapping = {old_index: new_index for new_index, old_index in enumerate(kept)}
    logging.info("源时间戳：%s（%d episodes）", source_dir, source_total)
    logging.info("删除：%s", sorted(deleted))
    logging.info("输出时间戳：%s（%d episodes）", output_dir, output_total)
    changed = [(old, new) for old, new in mapping.items() if old != new]
    if changed:
        logging.info("递补映射：%d->%d ... %d->%d", *changed[0], *changed[-1])
    else:
        logging.info("剩余 episode 序号无需递补")

    # Validate every source sidecar even in dry-run mode, so a successful dry run
    # proves both filenames and embedded episode_index values are consistent.
    for old_index, path in indexed_files.items():
        validate_internal_episode_index(path, old_index)

    if args.dry_run:
        logging.info("dry-run 检查通过，没有写入任何文件")
        return
    if output_dir.exists():
        raise FileExistsError(
            f"目标时间戳目录已存在，拒绝覆盖：{output_dir}。"
            "请先确认它是否包含需要保留的数据。"
        )

    staging_dir = output_root / f".ros_timestamps.tmp-{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=False)
    try:
        for old_index, new_index in mapping.items():
            output_path = staging_dir / f"episode_{new_index:06d}.parquet"
            rewrite_episode(indexed_files[old_index], output_path, old_index, new_index)
        staging_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    generated = indexed_timestamp_files(output_dir)
    if set(generated) != set(range(output_total)):
        raise RuntimeError("输出后校验失败：时间戳文件序号不连续")
    for index, path in generated.items():
        validate_internal_episode_index(path, index)
    logging.info("完成：生成 %d 个连续编号的 ROS 时间戳文件", output_total)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    synchronize_timestamps(parse_args())


if __name__ == "__main__":
    main()
