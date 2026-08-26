#!/usr/bin/env python3

"""Offline validation/test evaluation for Nero ACT checkpoints.

The evaluator follows the deployment inference path: observations are
normalized by the checkpoint preprocessor, ACT predicts without seeing the
ground-truth action, and predictions are unnormalized before physical-unit
errors are computed.  Depth is removed before samples are decoded.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


DEFAULT_CONFIG = Path(__file__).with_name("nero_ge_clean_act_train_config.json")
DEPTH_KEY = "observation.images.depth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--checkpoint",
        default="all",
        help="all、last、checkpoint step（如 100000）或 pretrained_model 路径",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--frame-stride", type=int, help="覆盖配置中的抽帧间隔")
    parser.add_argument("--max-samples-per-episode", type=int, help="0表示不限制")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只验证48/6/7数据划分和配置，不加载checkpoint、不解码视频",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open(encoding="utf-8") as stream:
        return json.load(stream)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了CUDA，但当前PyTorch无法使用CUDA")
    return requested


def validate_dataset(
    section_name: str,
    section: dict[str, Any],
    expected_robot_type: str,
) -> LeRobotDatasetMetadata:
    required = {"repo_id", "root", "expected_episodes", "expected_fps", "video_backend"}
    missing = required - set(section)
    if missing:
        raise ValueError(f"{section_name} 缺少配置：{sorted(missing)}")
    root = Path(section["root"]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{section_name}不存在：{root}")
    metadata = LeRobotDatasetMetadata(section["repo_id"], root=root)
    if metadata.robot_type != expected_robot_type:
        raise ValueError(
            f"{section_name} robot_type={metadata.robot_type!r}，应为{expected_robot_type!r}"
        )
    if metadata.total_episodes != int(section["expected_episodes"]):
        raise ValueError(
            f"{section_name}实际{metadata.total_episodes}轮，"
            f"配置要求{section['expected_episodes']}轮"
        )
    if metadata.fps != int(section["expected_fps"]):
        raise ValueError(
            f"{section_name}实际{metadata.fps} Hz，配置要求{section['expected_fps']} Hz"
        )
    return metadata


def validate_split_config(raw: dict[str, Any]) -> dict[str, Any]:
    required_sections = {
        "dataset",
        "validation_dataset",
        "test_dataset",
        "features",
        "training",
        "offline_validation",
    }
    missing = required_sections - set(raw)
    if missing:
        raise ValueError(f"配置缺少section：{sorted(missing)}")

    train_raw = raw["dataset"]
    train_metadata = LeRobotDatasetMetadata(
        train_raw["repo_id"], root=Path(train_raw["root"]).expanduser().resolve()
    )
    if train_metadata.total_episodes != 48:
        raise ValueError(f"训练集应为48轮，实际为{train_metadata.total_episodes}轮")
    if train_metadata.fps != int(train_raw["expected_fps"]):
        raise ValueError("训练集FPS与配置不一致")

    validation_metadata = validate_dataset(
        "validation_dataset", raw["validation_dataset"], train_raw["expected_robot_type"]
    )
    test_metadata = validate_dataset(
        "test_dataset", raw["test_dataset"], train_raw["expected_robot_type"]
    )

    train_features = train_metadata.features
    for name, metadata in (
        ("validation_dataset", validation_metadata),
        ("test_dataset", test_metadata),
    ):
        if set(metadata.features) != set(train_features):
            raise ValueError(f"{name}特征键与训练集不一致")
        mismatched = {
            key: (train_features[key]["shape"], metadata.features[key]["shape"])
            for key in train_features
            if train_features[key].get("shape") != metadata.features[key].get("shape")
        }
        if mismatched:
            raise ValueError(f"{name}特征尺寸与训练集不一致：{mismatched}")

    return {
        "train": {
            "root": str(train_metadata.root),
            "episodes": train_metadata.total_episodes,
            "frames": train_metadata.total_frames,
            "fps": train_metadata.fps,
        },
        "validation": {
            "root": str(validation_metadata.root),
            "episodes": validation_metadata.total_episodes,
            "frames": validation_metadata.total_frames,
            "fps": validation_metadata.fps,
        },
        "test": {
            "root": str(test_metadata.root),
            "episodes": test_metadata.total_episodes,
            "frames": test_metadata.total_frames,
            "fps": test_metadata.fps,
        },
    }


def checkpoint_paths(output_dir: Path, selection: str, split: str) -> list[Path]:
    checkpoints_dir = output_dir / "checkpoints"
    if selection == "all":
        if split == "test":
            raise ValueError("测试集禁止使用--checkpoint=all；请先用验证集选定一个checkpoint")
        paths = sorted(
            path / "pretrained_model"
            for path in checkpoints_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        ) if checkpoints_dir.is_dir() else []
    elif selection == "last":
        paths = [checkpoints_dir / "last" / "pretrained_model"]
    elif selection.isdigit():
        paths = [checkpoints_dir / f"{int(selection):06d}" / "pretrained_model"]
    else:
        candidate = Path(selection).expanduser().resolve()
        paths = [candidate / "pretrained_model" if (candidate / "pretrained_model").is_dir() else candidate]
    missing = [path for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"checkpoint不存在：{[str(path) for path in missing]}")
    if not paths:
        raise FileNotFoundError(f"没有找到checkpoint：{checkpoints_dir}")
    return paths


def checkpoint_step(path: Path) -> int:
    name = path.parent.resolve().name
    if not name.isdigit():
        raise ValueError(f"无法从checkpoint目录解析step：{path}")
    return int(name)


def make_dataset(
    section: dict[str, Any],
    policy_config: PreTrainedConfig,
    tolerance_s: float,
) -> LeRobotDataset:
    root = Path(section["root"]).expanduser().resolve()
    metadata = LeRobotDatasetMetadata(section["repo_id"], root=root)
    dataset = LeRobotDataset(
        section["repo_id"],
        root=root,
        delta_timestamps=resolve_delta_timestamps(policy_config, metadata),
        video_backend=section["video_backend"],
        tolerance_s=tolerance_s,
    )
    if DEPTH_KEY not in dataset.meta.features:
        raise ValueError(f"评测数据集缺少待排除特征{DEPTH_KEY}")
    dataset.meta.info["features"].pop(DEPTH_KEY)
    if dataset.meta.stats is not None:
        dataset.meta.stats.pop(DEPTH_KEY, None)
    if DEPTH_KEY in dataset.meta.camera_keys:
        raise RuntimeError("验证集depth仍会被解码")
    return dataset


def select_indices(
    dataset: LeRobotDataset,
    frame_stride: int,
    max_samples_per_episode: int,
) -> list[int]:
    episode_indices = dataset.hf_dataset["episode_index"]
    frame_indices = dataset.hf_dataset["frame_index"]
    selected: list[int] = []
    counts: dict[int, int] = defaultdict(int)
    for index, (episode, frame) in enumerate(zip(episode_indices, frame_indices, strict=True)):
        episode = int(episode)
        if int(frame) % frame_stride != 0:
            continue
        if max_samples_per_episode > 0 and counts[episode] >= max_samples_per_episode:
            continue
        selected.append(index)
        counts[episode] += 1
    if not selected:
        raise RuntimeError("评测抽样后没有任何帧")
    return selected


def evaluate_checkpoint(
    checkpoint: Path,
    dataset: LeRobotDataset,
    indices: list[int],
    device: str,
    action_prefix_steps: list[int],
) -> dict[str, Any]:
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if config.type != "act":
        raise ValueError(f"只支持ACT checkpoint，当前为{config.type!r}")
    config.device = device
    policy = ACTPolicy.from_pretrained(
        checkpoint, config=config, local_files_only=True
    ).to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    action_names = list(dataset.meta.features["action"]["names"])
    if len(action_names) != 8:
        raise ValueError(f"action维度不是8：{action_names}")
    first_raw_errors: list[np.ndarray] = []
    first_normalized_errors: list[np.ndarray] = []
    full_raw_mae: list[float] = []
    full_normalized_mae: list[float] = []
    prefix_raw_mae: dict[int, list[float]] = {step: [] for step in action_prefix_steps}
    episode_first_errors: dict[int, list[float]] = defaultdict(list)

    with torch.inference_mode():
        for index in tqdm(indices, desc=f"validate {checkpoint_step(checkpoint):06d}"):
            sample = dataset[index]
            target_raw = sample["action"].detach().cpu().float()
            is_pad = sample["action_is_pad"].detach().cpu().bool()
            processed = preprocessor(sample)
            target_normalized = processed["action"].detach().cpu().float()
            inference_batch = {key: processed[key] for key in config.input_features}
            predicted_normalized = policy.predict_action_chunk(inference_batch)
            predicted_raw = postprocessor(predicted_normalized).detach().cpu().float()[0]
            predicted_normalized = predicted_normalized.detach().cpu().float()[0]

            valid = ~is_pad
            if not bool(valid[0]):
                raise RuntimeError(f"样本{index}的第一个action被标记为padding")
            if predicted_raw.shape != target_raw.shape:
                raise RuntimeError(
                    f"预测/目标shape不一致：{predicted_raw.shape} != {target_raw.shape}"
                )

            raw_error = (predicted_raw - target_raw).abs()
            normalized_error = (predicted_normalized - target_normalized).abs()
            first_raw = raw_error[0].numpy()
            first_normalized = normalized_error[0].numpy()
            first_raw_errors.append(first_raw)
            first_normalized_errors.append(first_normalized)
            full_raw_mae.append(float(raw_error[valid].mean()))
            full_normalized_mae.append(float(normalized_error[valid].mean()))
            episode_first_errors[int(sample["episode_index"])].append(float(first_normalized.mean()))

            for prefix in action_prefix_steps:
                prefix_valid = valid[:prefix]
                if bool(prefix_valid.any()):
                    prefix_raw_mae[prefix].append(float(raw_error[:prefix][prefix_valid].mean()))

    first_raw_array = np.stack(first_raw_errors)
    first_normalized_array = np.stack(first_normalized_errors)
    episode_means = np.asarray(
        [np.mean(values) for _, values in sorted(episode_first_errors.items())], dtype=np.float64
    )
    result: dict[str, Any] = {
        "step": checkpoint_step(checkpoint),
        "checkpoint": str(checkpoint),
        "sample_count": len(indices),
        "episode_count": len(episode_means),
        "first_action_normalized_mae": float(first_normalized_array.mean()),
        "first_action_normalized_p95": float(np.percentile(first_normalized_array, 95)),
        "episode_balanced_normalized_mae_mean": float(episode_means.mean()),
        "episode_balanced_normalized_mae_std": float(episode_means.std()),
        "episode_worst_normalized_mae": float(episode_means.max()),
        "first_action_raw_mae": float(first_raw_array.mean()),
        "first_action_raw_p95": float(np.percentile(first_raw_array, 95)),
        "first_action_arm_mae_rad": float(first_raw_array[:, :7].mean()),
        "first_action_gripper_mae": float(first_raw_array[:, 7].mean()),
        "full_chunk_raw_mae": float(np.mean(full_raw_mae)),
        "full_chunk_normalized_mae": float(np.mean(full_normalized_mae)),
    }
    for prefix, values in prefix_raw_mae.items():
        result[f"prefix_{prefix}_raw_mae"] = float(np.mean(values))
    for dimension, name in enumerate(action_names):
        safe_name = name.replace(".", "_")
        result[f"first_{safe_name}_mae"] = float(first_raw_array[:, dimension].mean())
        result[f"first_{safe_name}_p95"] = float(
            np.percentile(first_raw_array[:, dimension], 95)
        )

    del policy, preprocessor, postprocessor
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(rows: list[dict[str, Any]], output_path: Path, split: str) -> None:
    steps = [int(row["step"]) for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].plot(steps, [row["first_action_normalized_mae"] for row in rows], marker="o")
    axes[0, 0].set_title("First-action normalized MAE")
    axes[0, 1].plot(steps, [row["first_action_arm_mae_rad"] for row in rows], marker="o")
    axes[0, 1].set_title("First-action arm MAE (rad)")
    axes[1, 0].plot(steps, [row["first_action_gripper_mae"] for row in rows], marker="o")
    axes[1, 0].set_title("First-action gripper MAE")
    for key in sorted(
        (key for key in rows[0] if key.startswith("prefix_") and key.endswith("_raw_mae")),
        key=lambda value: int(value.split("_")[1]),
    ):
        axes[1, 1].plot(steps, [row[key] for row in rows], marker="o", label=key.split("_")[1])
    axes[1, 1].set_title("Action-prefix raw MAE")
    axes[1, 1].legend(title="prefix steps")
    for axis in axes.flat:
        axis.set_xlabel("Checkpoint step")
        axis.grid(True, alpha=0.25)
    figure.suptitle(f"Nero ACT {split} checkpoint metrics")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config_path = args.config.expanduser().resolve()
    raw = load_json(config_path)
    split_report = validate_split_config(raw)
    print(json.dumps(split_report, ensure_ascii=False, indent=2))
    if args.validate_only:
        return

    offline = raw["offline_validation"]
    frame_stride = args.frame_stride if args.frame_stride is not None else int(offline["frame_stride"])
    max_samples = (
        args.max_samples_per_episode
        if args.max_samples_per_episode is not None
        else int(offline["max_samples_per_episode"])
    )
    if frame_stride <= 0 or max_samples < 0:
        raise ValueError("frame_stride必须大于0，max_samples_per_episode不能小于0")
    prefixes = sorted({int(value) for value in offline["action_prefix_steps"]})
    if not prefixes or prefixes[0] <= 0:
        raise ValueError("action_prefix_steps必须为正整数")

    device = resolve_device(args.device if args.device != "auto" else offline["device"])
    training_output = Path(raw["training"]["output_dir"]).expanduser().resolve()
    checkpoints = checkpoint_paths(training_output, args.checkpoint, args.split)
    first_config = PreTrainedConfig.from_pretrained(checkpoints[0], local_files_only=True)
    if max(prefixes) > int(first_config.chunk_size):
        raise ValueError("action_prefix_steps超过checkpoint chunk_size")
    section = raw[f"{args.split}_dataset"]
    dataset = make_dataset(section, first_config, float(raw["training"]["tolerance_s"]))
    indices = select_indices(dataset, frame_stride, max_samples)
    logging.info(
        "%s：%d轮、%d帧，按stride=%d选取%d帧，depth未解码",
        args.split,
        dataset.num_episodes,
        dataset.num_frames,
        frame_stride,
        len(indices),
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else training_output / f"{args.split}_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        row = evaluate_checkpoint(checkpoint, dataset, indices, device, prefixes)
        rows.append(row)
        rows.sort(key=lambda value: int(value["step"]))
        write_csv(output_dir / f"{args.split}_checkpoint_metrics.csv", rows)
        logging.info(
            "step=%d normalized_MAE=%.6f arm_MAE=%.6f gripper_MAE=%.6f",
            row["step"],
            row["first_action_normalized_mae"],
            row["first_action_arm_mae_rad"],
            row["first_action_gripper_mae"],
        )

    best = min(rows, key=lambda value: value["episode_balanced_normalized_mae_mean"])
    summary = {
        "config": str(config_path),
        "split": args.split,
        "dataset": split_report[args.split],
        "device": device,
        "frame_stride": frame_stride,
        "evaluated_samples": len(indices),
        "checkpoint_count": len(rows),
        "selection_metric": "episode_balanced_normalized_mae_mean",
        "best_checkpoint_step": int(best["step"]),
        "best_metric": float(best["episode_balanced_normalized_mae_mean"]),
        "test_set_policy": "test split requires one preselected checkpoint",
    }
    (output_dir / f"{args.split}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_metrics(rows, output_dir / f"{args.split}_checkpoint_curves.png", args.split)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
