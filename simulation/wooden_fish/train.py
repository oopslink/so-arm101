"""CPU PPO baseline. The default is deliberately a small smoke run."""
import argparse
from dataclasses import asdict
from functools import partial
import importlib.metadata
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from .env import WoodenFishEnv, TaskConfig
from .scene import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.envs < 1:
        parser.error("steps and envs must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = TaskConfig()
    if args.resume:
        previous = json.loads((args.resume.parent / "config.json").read_text())
        config = TaskConfig.from_saved(previous["task"])
    env = make_vec_env(partial(WoodenFishEnv, config=config), n_envs=args.envs,
                       seed=args.seed, monitor_dir=str(args.output / "monitor"),
                       vec_env_cls=SubprocVecEnv if args.envs > 1 else DummyVecEnv)
    evaluation = make_vec_env(partial(WoodenFishEnv, config=config), seed=10000,
                               vec_env_cls=SubprocVecEnv if args.envs > 1 else DummyVecEnv)
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
        record = {"task": asdict(config), "seed": args.seed, "envs": args.envs,
                  "requested_steps": args.steps, "python": sys.version,
                  "git_revision": revision, "git_status": dirty,
                  "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(ROOT.rglob("*"))
                                    if p.is_file() and ("wooden_fish" in p.parts or "assets" in p.parts)
                                    and "__pycache__" not in p.parts},
                  "resume": str(args.resume) if args.resume else None,
                  "versions": {p: importlib.metadata.version(p) for p in
                               ("mujoco", "gymnasium", "stable-baselines3", "torch", "numpy")}}
        (args.output / "config.json").write_text(json.dumps(record, indent=2) + "\n")
        if args.resume:
            model = PPO.load(args.resume, env=env, device="cpu", tensorboard_log=str(args.output / "tensorboard"))
        else:
            model = PPO("MlpPolicy", env, device="cpu", seed=args.seed, n_steps=256,
                        batch_size=64, n_epochs=5, learning_rate=3e-4,
                        policy_kwargs={"net_arch": [64, 64]}, verbose=1,
                        tensorboard_log=str(args.output / "tensorboard"))
        callbacks = [
            CheckpointCallback(save_freq=max(2048 // args.envs, 1), save_path=str(args.output), name_prefix="checkpoint"),
            EvalCallback(evaluation, best_model_save_path=str(args.output),
                         log_path=str(args.output / "eval"),
                         eval_freq=max(2048 // args.envs, 1), n_eval_episodes=5),
        ]
        model.learn(total_timesteps=args.steps, callback=callbacks, reset_num_timesteps=not bool(args.resume))
        model.save(args.output / "final_model")
    finally:
        env.close()
        evaluation.close()


if __name__ == "__main__":
    main()
