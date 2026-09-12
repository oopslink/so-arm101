#!/usr/bin/env python3
"""Evaluate a local PI0 checkpoint on recorded LeRobot dataset frames.

This is an offline diagnostic: it never connects to a robot or writes motor
commands.  For each selected frame it predicts one action chunk and compares
that chunk with the recorded actions that followed the same observation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.rollout.context import _load_pretrained_policy


JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def episode_bounds(dataset: LeRobotDataset, episode: int) -> tuple[int, int]:
    row = dataset.meta.episodes[episode]
    return int(row["dataset_from_index"]), int(row["dataset_to_index"])


def automatic_frames(dataset: LeRobotDataset, episode: int) -> list[int]:
    """Choose timeline points plus gripper-open/close transitions."""
    start, end = episode_bounds(dataset, episode)
    length = end - start
    frames = {round(ratio * (length - 1)) for ratio in (0, .2, .3, .4, .5, .6, .7, .8, .9)}

    gripper = np.asarray([float(dataset[index]["action"][-1]) for index in range(start, end)])
    baseline = float(np.median(gripper[: min(30, length)]))
    open_indexes = np.flatnonzero(gripper > baseline + 5.0)
    if open_indexes.size:
        first_open = int(open_indexes[0])
        peak_open = int(np.argmax(gripper))
        frames.update((max(0, first_open - 15), first_open, peak_open))
        close_indexes = np.flatnonzero(
            (np.arange(length) > peak_open) & (gripper < baseline + 2.0)
        )
        if close_indexes.size:
            frames.add(int(close_indexes[0]))
    return sorted(frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repo-id", default="local/qiaomuyu")
    parser.add_argument("--episodes", type=parse_int_list, default=[0])
    parser.add_argument("--frames", type=parse_int_list)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    tokenizer_path = model_path / "tokenizer"
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Checkpoint tokenizer is missing: {tokenizer_path}")

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.dataset)
    config = PreTrainedConfig.from_pretrained(str(model_path))
    # from_pretrained preserves the base checkpoint path in config.json.  Force
    # weight loading from the fine-tuned local checkpoint.
    config.pretrained_path = str(model_path)
    config.device = args.device

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(model_path),
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            # The exported config says "./tokenizer". Resolve it against the
            # checkpoint, not the caller's current working directory.
            "tokenizer_processor": {"tokenizer_name": str(tokenizer_path)},
        },
    )
    policy = _load_pretrained_policy(config).to(args.device).eval()

    results: list[dict[str, object]] = []
    for episode in args.episodes:
        start, end = episode_bounds(dataset, episode)
        frames = args.frames if args.frames is not None else automatic_frames(dataset, episode)
        for frame in frames:
            if not 0 <= frame < end - start:
                continue
            sample = dataset[start + frame]
            observation = {
                key: value
                for key, value in sample.items()
                if key.startswith("observation.")
            }
            observation["task"] = sample["task"]
            observation["robot_type"] = "so_follower"

            if hasattr(preprocessor, "reset"):
                preprocessor.reset()
            if hasattr(postprocessor, "reset"):
                postprocessor.reset()
            policy.reset()
            torch.manual_seed(args.seed)
            batch = preprocessor(observation)
            predicted = postprocessor(policy.predict_action_chunk(batch))[0].detach().cpu()

            usable = min(int(predicted.shape[0]), end - (start + frame))
            truth = torch.stack(
                [dataset[start + frame + offset]["action"] for offset in range(usable)]
            )
            predicted = predicted[:usable]
            error = (predicted - truth).abs()
            row = {
                "episode": episode,
                "frame": frame,
                "seconds": round(frame / dataset.fps, 3),
                "state": [round(float(v), 3) for v in sample["observation.state"]],
                "truth_first": [round(float(v), 3) for v in truth[0]],
                "predicted_first": [round(float(v), 3) for v in predicted[0]],
                "truth_gripper_range": [
                    round(float(truth[:, -1].min()), 3),
                    round(float(truth[:, -1].max()), 3),
                ],
                "predicted_gripper_range": [
                    round(float(predicted[:, -1].min()), 3),
                    round(float(predicted[:, -1].max()), 3),
                ],
                "mae_by_joint": {
                    name: round(float(error[:, index].mean()), 3)
                    for index, name in enumerate(JOINTS)
                },
                "mae_all": round(float(error.mean()), 3),
            }
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = {
        "model": str(model_path),
        "dataset": str(Path(args.dataset).resolve()),
        "samples": len(results),
        "mean_mae": round(float(np.mean([row["mae_all"] for row in results])), 3),
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
