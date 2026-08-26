#!/usr/bin/env python3

"""Train ACT on a Nero ROS 2 dataset without using the depth camera.

All dataset, feature, policy, optimizer and training parameters live in the
adjacent JSON file.  The script validates the real LeRobot metadata, supplies
only state/color/fisheye to ACT, and removes depth from the in-memory dataset
view so the DataLoader does not decode the depth MP4.  Source dataset files are
never modified.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterator

from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.transforms import ImageTransformsConfig
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.optim.optimizers import AdamWConfig
from lerobot.policies.act.configuration_act import ACTConfig


DEFAULT_CONFIG_PATH = Path(__file__).with_name("nero_ros2_act_train_config.json")
DEPTH_KEY = "observation.images.depth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 state + RGB + fisheye 训练 Nero ACT，明确排除深度图。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"独立 JSON 配置文件（默认：{DEFAULT_CONFIG_PATH}）",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="检查数据、特征和一条实际样本，不创建模型、不开始训练",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "从官方checkpoint完整续训，可传 checkpoints/last、数字checkpoint目录、"
            "pretrained_model目录或其中的train_config.json"
        ),
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"训练配置不存在：{path}")
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    required_sections = {
        "dataset",
        "features",
        "policy",
        "optimizer",
        "training",
        "wandb",
        "evaluation",
    }
    missing = required_sections - set(config)
    if missing:
        raise ValueError(f"配置文件缺少 section：{sorted(missing)}")
    return config


def require_keys(section_name: str, section: dict[str, Any], keys: set[str]) -> None:
    missing = keys - set(section)
    if missing:
        raise ValueError(f"配置 {section_name} 缺少参数：{sorted(missing)}")


def validate_dataset_and_features(
    raw: dict[str, Any],
) -> tuple[LeRobotDatasetMetadata, dict[str, Any], dict[str, Any], list[str]]:
    dataset_cfg = raw["dataset"]
    feature_cfg = raw["features"]
    require_keys(
        "dataset",
        dataset_cfg,
        {
            "repo_id",
            "root",
            "expected_robot_type",
            "expected_fps",
            "episodes",
            "revision",
            "video_backend",
            "streaming",
            "use_imagenet_stats",
            "image_transforms",
        },
    )
    require_keys("features", feature_cfg, {"input", "output", "exclude"})

    root = Path(dataset_cfg["root"]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"数据集不存在：{root}\n"
            "如果你要训练同步后的数据，请先把配置 dataset.root 改为同步输出目录。"
        )
    metadata = LeRobotDatasetMetadata(
        dataset_cfg["repo_id"],
        root=root,
        revision=dataset_cfg["revision"],
    )
    if metadata.robot_type != dataset_cfg["expected_robot_type"]:
        raise ValueError(
            f"robot_type 不一致：数据集={metadata.robot_type!r}，"
            f"配置={dataset_cfg['expected_robot_type']!r}"
        )
    if metadata.fps != int(dataset_cfg["expected_fps"]):
        raise ValueError(
            f"FPS 不一致：数据集={metadata.fps}，配置={dataset_cfg['expected_fps']}。"
            "同步到新 FPS 后必须同时更新 expected_fps 和 ACT chunk_size。"
        )

    all_features = dataset_to_policy_features(metadata.features)
    input_names = list(feature_cfg["input"])
    output_names = list(feature_cfg["output"])
    excluded = list(feature_cfg["exclude"])
    if len(input_names) != len(set(input_names)):
        raise ValueError(f"features.input 存在重复项：{input_names}")
    if len(output_names) != len(set(output_names)):
        raise ValueError(f"features.output 存在重复项：{output_names}")
    if len(excluded) != len(set(excluded)):
        raise ValueError(f"features.exclude 存在重复项：{excluded}")
    missing_features = (set(input_names) | set(output_names) | set(excluded)) - set(all_features)
    if missing_features:
        raise ValueError(f"数据集缺少配置中的特征：{sorted(missing_features)}")
    overlap = (set(input_names) | set(output_names)) & set(excluded)
    if overlap:
        raise ValueError(f"特征不能同时使用和排除：{sorted(overlap)}")
    if DEPTH_KEY not in excluded:
        raise ValueError(f"本脚本要求 features.exclude 明确包含 {DEPTH_KEY!r}")
    if DEPTH_KEY in input_names:
        raise ValueError("深度图不能出现在 ACT input features 中")
    if output_names != ["action"]:
        raise ValueError(f"Nero ACT 输出必须只包含 action，当前为：{output_names}")

    input_features = {name: all_features[name] for name in input_names}
    output_features = {name: all_features[name] for name in output_names}
    visual_shapes = {
        feature.shape
        for feature in input_features.values()
        if feature.type is FeatureType.VISUAL
    }
    if not visual_shapes:
        raise ValueError("ACT 至少需要一项视觉输入")
    if len(visual_shapes) != 1:
        raise ValueError(f"ACT 的多相机图像尺寸必须一致，当前为：{sorted(visual_shapes)}")
    if "observation.state" not in input_features:
        raise ValueError("Nero ACT 输入必须包含 observation.state")
    if input_features["observation.state"].shape != (8,):
        raise ValueError(
            f"observation.state 应为 7 关节 + 1 夹爪，实际为 "
            f"{input_features['observation.state'].shape}"
        )
    if output_features["action"].shape != (8,):
        raise ValueError(
            f"action 应为 7 关节 + 1 夹爪，实际为 {output_features['action'].shape}"
        )

    requested_episodes = dataset_cfg["episodes"]
    if requested_episodes is not None:
        if not isinstance(requested_episodes, list) or not all(
            isinstance(index, int) for index in requested_episodes
        ):
            raise ValueError("dataset.episodes 必须是 null 或整数数组")
        invalid = {
            index
            for index in requested_episodes
            if index < 0 or index >= metadata.total_episodes
        }
        if invalid:
            raise ValueError(
                f"dataset.episodes 包含不存在的轮次：{sorted(invalid)}；"
                f"数据集共 {metadata.total_episodes} 轮"
            )

    return metadata, input_features, output_features, excluded


def make_pipeline_config(
    raw: dict[str, Any],
    input_features: dict[str, Any],
    output_features: dict[str, Any],
) -> TrainPipelineConfig:
    dataset_raw = raw["dataset"]
    policy_raw = raw["policy"]
    optimizer_raw = raw["optimizer"]
    training_raw = raw["training"]
    wandb_raw = raw["wandb"]
    eval_raw = raw["evaluation"]

    require_keys(
        "policy",
        policy_raw,
        {
            "device",
            "use_amp",
            "use_peft",
            "pretrained_path",
            "push_to_hub",
            "repo_id",
            "private",
            "license",
            "tags",
            "n_obs_steps",
            "chunk_size",
            "n_action_steps",
            "normalization_mapping",
            "vision_backbone",
            "pretrained_backbone_weights",
            "replace_final_stride_with_dilation",
            "pre_norm",
            "dim_model",
            "n_heads",
            "dim_feedforward",
            "feedforward_activation",
            "n_encoder_layers",
            "n_decoder_layers",
            "use_vae",
            "latent_dim",
            "n_vae_encoder_layers",
            "temporal_ensemble_coeff",
            "dropout",
            "kl_weight",
        },
    )
    require_keys(
        "optimizer",
        optimizer_raw,
        {
            "type",
            "lr",
            "backbone_lr",
            "weight_decay",
            "betas",
            "eps",
            "grad_clip_norm",
            "scheduler",
        },
    )
    if optimizer_raw["type"] != "adamw":
        raise ValueError("Nero ACT 训练脚本当前只支持 optimizer.type=adamw")
    if optimizer_raw["scheduler"] is not None:
        raise ValueError("ACT 当前配置只支持 optimizer.scheduler=null")
    if len(optimizer_raw["betas"]) != 2:
        raise ValueError("optimizer.betas 必须包含两个数")

    normalization_mapping = {
        feature_type: NormalizationMode(mode)
        for feature_type, mode in policy_raw["normalization_mapping"].items()
    }
    pretrained_path = policy_raw["pretrained_path"]
    policy = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        device=policy_raw["device"],
        use_amp=bool(policy_raw["use_amp"]),
        use_peft=bool(policy_raw["use_peft"]),
        pretrained_path=Path(pretrained_path) if pretrained_path else None,
        push_to_hub=bool(policy_raw["push_to_hub"]),
        repo_id=policy_raw["repo_id"],
        private=policy_raw["private"],
        license=policy_raw["license"],
        tags=list(policy_raw["tags"]),
        n_obs_steps=int(policy_raw["n_obs_steps"]),
        chunk_size=int(policy_raw["chunk_size"]),
        n_action_steps=int(policy_raw["n_action_steps"]),
        normalization_mapping=normalization_mapping,
        vision_backbone=policy_raw["vision_backbone"],
        pretrained_backbone_weights=policy_raw["pretrained_backbone_weights"],
        replace_final_stride_with_dilation=bool(
            policy_raw["replace_final_stride_with_dilation"]
        ),
        pre_norm=bool(policy_raw["pre_norm"]),
        dim_model=int(policy_raw["dim_model"]),
        n_heads=int(policy_raw["n_heads"]),
        dim_feedforward=int(policy_raw["dim_feedforward"]),
        feedforward_activation=policy_raw["feedforward_activation"],
        n_encoder_layers=int(policy_raw["n_encoder_layers"]),
        n_decoder_layers=int(policy_raw["n_decoder_layers"]),
        use_vae=bool(policy_raw["use_vae"]),
        latent_dim=int(policy_raw["latent_dim"]),
        n_vae_encoder_layers=int(policy_raw["n_vae_encoder_layers"]),
        temporal_ensemble_coeff=policy_raw["temporal_ensemble_coeff"],
        dropout=float(policy_raw["dropout"]),
        kl_weight=float(policy_raw["kl_weight"]),
        optimizer_lr=float(optimizer_raw["lr"]),
        optimizer_lr_backbone=float(optimizer_raw["backbone_lr"]),
        optimizer_weight_decay=float(optimizer_raw["weight_decay"]),
    )

    # Preserve ACT's two optimizer parameter groups while making every AdamW
    # parameter, including gradient clipping, come from the JSON file.
    def explicit_optimizer_preset(_self: ACTConfig) -> AdamWConfig:
        return AdamWConfig(
            lr=float(optimizer_raw["lr"]),
            betas=tuple(float(value) for value in optimizer_raw["betas"]),
            eps=float(optimizer_raw["eps"]),
            weight_decay=float(optimizer_raw["weight_decay"]),
            grad_clip_norm=float(optimizer_raw["grad_clip_norm"]),
        )

    policy.get_optimizer_preset = MethodType(explicit_optimizer_preset, policy)

    transform_raw = dataset_raw["image_transforms"]
    require_keys(
        "dataset.image_transforms",
        transform_raw,
        {"enable", "max_num_transforms", "random_order"},
    )
    dataset = DatasetConfig(
        repo_id=dataset_raw["repo_id"],
        root=str(Path(dataset_raw["root"]).expanduser().resolve()),
        episodes=dataset_raw["episodes"],
        image_transforms=ImageTransformsConfig(
            enable=bool(transform_raw["enable"]),
            max_num_transforms=int(transform_raw["max_num_transforms"]),
            random_order=bool(transform_raw["random_order"]),
        ),
        revision=dataset_raw["revision"],
        use_imagenet_stats=bool(dataset_raw["use_imagenet_stats"]),
        video_backend=dataset_raw["video_backend"],
        streaming=bool(dataset_raw["streaming"]),
    )
    wandb = WandBConfig(
        enable=bool(wandb_raw["enable"]),
        disable_artifact=bool(wandb_raw["disable_artifact"]),
        project=wandb_raw["project"],
        entity=wandb_raw["entity"],
        notes=wandb_raw["notes"],
        run_id=wandb_raw["run_id"],
        mode=wandb_raw["mode"],
    )
    evaluation = EvalConfig(
        n_episodes=int(eval_raw["n_episodes"]),
        batch_size=int(eval_raw["batch_size"]),
        use_async_envs=bool(eval_raw["use_async_envs"]),
    )
    if bool(training_raw["resume"]):
        raise ValueError(
            "此独立配置用于新训练，training.resume 必须为 false；"
            "恢复训练请使用 checkpoint 中官方生成的 train_config.json。"
        )

    return TrainPipelineConfig(
        dataset=dataset,
        env=None,
        policy=policy,
        output_dir=Path(training_raw["output_dir"]).expanduser().resolve(),
        job_name=training_raw["job_name"],
        resume=False,
        seed=int(training_raw["seed"]),
        num_workers=int(training_raw["num_workers"]),
        batch_size=int(training_raw["batch_size"]),
        steps=int(training_raw["steps"]),
        eval_freq=int(training_raw["eval_freq"]),
        log_freq=int(training_raw["log_freq"]),
        tolerance_s=float(training_raw["tolerance_s"]),
        save_checkpoint=bool(training_raw["save_checkpoint"]),
        save_freq=int(training_raw["save_freq"]),
        use_policy_training_preset=True,
        optimizer=None,
        scheduler=None,
        eval=evaluation,
        wandb=wandb,
        peft=None,
        use_rabc=bool(training_raw["use_rabc"]),
        rename_map={},
    )


def remove_excluded_features_in_memory(
    dataset: LeRobotDataset, excluded: list[str]
) -> LeRobotDataset:
    """Hide excluded modalities from decoding without changing disk metadata."""
    for feature_name in excluded:
        if feature_name not in dataset.meta.features:
            raise RuntimeError(f"过滤前数据集缺少特征：{feature_name}")
        dataset.meta.info["features"].pop(feature_name)
        if dataset.meta.stats is not None:
            dataset.meta.stats.pop(feature_name, None)
    leaked = set(excluded) & set(dataset.meta.camera_keys)
    if leaked:
        raise AssertionError(f"被排除的相机仍会被解码：{sorted(leaked)}")
    return dataset


@contextmanager
def filtered_dataset_factory(
    train_module: Any,
    excluded: list[str],
) -> Iterator[None]:
    """Temporarily replace only this process's training dataset factory."""
    original_factory: Callable[..., LeRobotDataset] = train_module.make_dataset

    def make_filtered_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset:
        dataset = original_factory(cfg)
        return remove_excluded_features_in_memory(dataset, excluded)

    train_module.make_dataset = make_filtered_dataset
    try:
        yield
    finally:
        train_module.make_dataset = original_factory


def validate_one_sample(
    train_module: Any,
    cfg: TrainPipelineConfig,
    excluded: list[str],
) -> dict[str, Any]:
    cfg.validate()
    with filtered_dataset_factory(train_module, excluded):
        dataset = train_module.make_dataset(cfg)
        sample = dataset[0]
    leaked = set(excluded) & set(sample)
    if leaked:
        raise AssertionError(f"实际训练样本仍包含排除特征：{sorted(leaked)}")
    image_shapes = {
        key: list(sample[key].shape)
        for key in cfg.policy.input_features
        if cfg.policy.input_features[key].type is FeatureType.VISUAL
    }
    return {
        "dataset_root": cfg.dataset.root,
        "dataset_fps": dataset.fps,
        "dataset_episodes": dataset.num_episodes,
        "dataset_frames": dataset.num_frames,
        "policy_input_features": list(cfg.policy.input_features),
        "policy_output_features": list(cfg.policy.output_features),
        "decoded_camera_keys": dataset.meta.camera_keys,
        "excluded_features": excluded,
        "depth_present_in_sample": DEPTH_KEY in sample,
        "sample_image_shapes": image_shapes,
        "action_sample_shape": list(sample["action"].shape),
        "requested_device": raw_device_from_policy(cfg),
        "resolved_device": cfg.policy.device,
        "output_dir": str(cfg.output_dir),
    }


def raw_device_from_policy(cfg: TrainPipelineConfig) -> str | None:
    # ACTConfig may fall back when a requested accelerator is unavailable.
    return getattr(cfg.policy, "_nero_requested_device", cfg.policy.device)


def resolve_resume_paths(value: Path) -> tuple[Path, Path]:
    """Return ``(checkpoint_dir, train_config_path)`` for a resume argument."""
    path = value.expanduser().resolve()
    if path.is_file():
        if path.name != "train_config.json" or path.parent.name != "pretrained_model":
            raise ValueError("--resume-from文件必须是 pretrained_model/train_config.json")
        checkpoint_dir = path.parent.parent
        train_config_path = path
    elif path.is_dir() and path.name == "pretrained_model":
        checkpoint_dir = path.parent
        train_config_path = path / "train_config.json"
    elif path.is_dir():
        checkpoint_dir = path
        train_config_path = path / "pretrained_model" / "train_config.json"
    else:
        raise FileNotFoundError(f"续训checkpoint不存在：{path}")
    if not train_config_path.is_file():
        raise FileNotFoundError(f"checkpoint缺少train_config.json：{train_config_path}")
    training_state = checkpoint_dir / "training_state"
    required_state = {
        "training_step.json",
        "optimizer_state.safetensors",
        "optimizer_param_groups.json",
        "rng_state.safetensors",
    }
    missing = sorted(name for name in required_state if not (training_state / name).is_file())
    if missing:
        raise FileNotFoundError(f"checkpoint训练状态不完整，缺少：{missing}")
    return checkpoint_dir, train_config_path


def make_resume_config(
    raw: dict[str, Any],
    fresh_cfg: TrainPipelineConfig,
    resume_from: Path,
) -> tuple[TrainPipelineConfig, Path]:
    checkpoint_dir, train_config_path = resolve_resume_paths(resume_from)
    cfg = TrainPipelineConfig.from_pretrained(train_config_path, cli_args=["--resume=true"])
    expected_output = fresh_cfg.output_dir.resolve()
    resumed_output = cfg.output_dir.resolve()
    if resumed_output != expected_output:
        raise ValueError(
            f"checkpoint output_dir={resumed_output}，当前配置output_dir={expected_output}，拒绝混用"
        )
    expected_root = Path(fresh_cfg.dataset.root).resolve()
    resumed_root = Path(cfg.dataset.root).resolve()
    if resumed_root != expected_root:
        raise ValueError(
            f"checkpoint训练集={resumed_root}，当前配置训练集={expected_root}，拒绝混用"
        )
    target_steps = int(raw["training"]["steps"])
    training_step_path = checkpoint_dir / "training_state" / "training_step.json"
    with training_step_path.open(encoding="utf-8") as stream:
        saved_step_raw = json.load(stream)
    if isinstance(saved_step_raw, dict):
        saved_step = int(saved_step_raw.get("step", saved_step_raw.get("training_step", -1)))
    else:
        saved_step = int(saved_step_raw)
    if saved_step < 0:
        raise ValueError(f"无法读取checkpoint step：{training_step_path}")
    if target_steps <= saved_step:
        raise ValueError(
            f"checkpoint已经到step={saved_step}，当前配置目标steps={target_steps}；"
            "继续训练时请把配置中的training.steps设为更大的最终总步数"
        )
    if int(cfg.batch_size) != int(fresh_cfg.batch_size):
        raise ValueError(
            f"checkpoint batch_size={cfg.batch_size}，当前配置batch_size={fresh_cfg.batch_size}，"
            "完整续训期间不能改变batch_size"
        )
    # Model/optimizer/dataset settings come from the checkpoint.  Only extend
    # the total target and adopt non-stateful logging/checkpoint intervals from
    # the current explicit JSON configuration.
    cfg.steps = target_steps
    cfg.log_freq = fresh_cfg.log_freq
    cfg.save_checkpoint = fresh_cfg.save_checkpoint
    cfg.save_freq = fresh_cfg.save_freq
    cfg.eval_freq = fresh_cfg.eval_freq
    cfg.resume = True
    cfg.checkpoint_path = checkpoint_dir
    cfg.policy.pretrained_path = checkpoint_dir / "pretrained_model"
    cfg.policy._nero_requested_device = raw["policy"]["device"]
    return cfg, train_config_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raw = load_config(args.config)
    metadata, input_features, output_features, excluded = validate_dataset_and_features(raw)
    cfg = make_pipeline_config(raw, input_features, output_features)
    cfg.policy._nero_requested_device = raw["policy"]["device"]

    resume_train_config = None
    if args.resume_from is not None:
        if args.validate_only:
            raise ValueError("--validate-only和--resume-from不能同时使用")
        cfg, resume_train_config = make_resume_config(raw, cfg, args.resume_from)

    logging.info(
        "Nero dataset: %d episodes, %d frames, %d Hz",
        metadata.total_episodes,
        metadata.total_frames,
        metadata.fps,
    )
    logging.info("ACT inputs: %s", list(input_features))
    logging.info("ACT outputs: %s", list(output_features))
    logging.info("Excluded from decoding: %s", excluded)

    import lerobot.scripts.lerobot_train as train_module

    if args.validate_only:
        report = validate_one_sample(train_module, cfg, excluded)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    original_argv = sys.argv[:]
    try:
        if resume_train_config is not None:
            # TrainPipelineConfig.validate() intentionally re-reads config_path
            # to locate the checkpoint training_state directory.
            sys.argv = [
                original_argv[0],
                f"--config_path={resume_train_config}",
                "--resume=true",
            ]
            logging.info("完整续训：%s", cfg.checkpoint_path)
        with filtered_dataset_factory(train_module, excluded):
            train_module.train(cfg)
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
