"""Run a scripted baseline or evaluate a saved PPO checkpoint."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from .env import WoodenFishEnv, TaskConfig


def scripted_action(env):
    # Closed-loop positional IK only supplies targets; step() still runs servo physics.
    height = 0.045 if env.detector.count else -0.004
    target = env.target + [0, 0, env.head_radius + height]
    joints = env.solve_ik(target, seed=env.data.qpos.copy())
    return np.clip((joints[:5] - env.data.ctrl[:5]) / env.config.action_scale, -1, 1)


def evaluate_contact(args):
    from .grasp import ContactGrasp, rollout
    if args.fixed:
        raise ValueError("--fixed belongs to the historical --fixed-grip baseline")
    writer = None
    results = []
    try:
        if args.video:
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(args.video, fps=50)
        for episode in range(args.episodes):
            sim = ContactGrasp(seed=args.seed+episode)
            try:
                for event in rollout(sim):
                    if isinstance(event, dict):
                        report = {"seed": args.seed+episode, "is_success": event["passed"], **event}
                        results.append(report)
                        print(json.dumps(report), flush=True)
                    elif writer:
                        writer.append_data(sim.render(args.camera))
            finally:
                sim.close()
    finally:
        if writer:
            writer.close()
    report = {"controller": "contact_grasp_script", "success_rate": sum(r["passed"] for r in results)/len(results),
              "episodes": results, "note": "Starts with rod between fingers; not tabletop pickup or RL."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(f"success_rate={report['success_rate']:.1%}; report={args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--camera", choices=["overview", "side", "wrist"], default="overview")
    parser.add_argument("--fixed-grip", action="store_true", help="Use historical attached-mallet scripted baseline")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if not args.model and not args.fixed_grip:
        evaluate_contact(args)
        return
    config = TaskConfig(target_jitter=0, start_jitter=0) if args.fixed else TaskConfig()
    policy = None
    if args.model:
        from stable_baselines3 import PPO
        # Keep evaluation semantics identical to the checkpoint's recorded task.
        settings = args.model.parent / "config.json"
        if not settings.exists():
            parser.error("model directory must contain its training config.json")
        config = TaskConfig.from_saved(json.loads(settings.read_text())["task"])
        if args.fixed:
            config.target_jitter = config.start_jitter = 0
        policy = PPO.load(args.model, device="cpu")
    env = WoodenFishEnv(config=config)
    writer = None
    results = []
    try:
        if args.video:
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(args.video, fps=env.metadata["render_fps"])
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            total = 0.0
            if writer:
                writer.append_data(env.render(camera=None if args.camera == "overview" else args.camera))
            for _ in range(config.max_steps):
                action = policy.predict(obs, deterministic=True)[0] if policy else scripted_action(env)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                if writer:
                    writer.append_data(env.render(camera=None if args.camera == "overview" else args.camera))
                if terminated or truncated:
                    break
            result = {"seed": args.seed + episode, "return": total, **info}
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        if writer:
            writer.close()
        env.close()
    report = {"controller": str(args.model) if policy else "scripted_ik",
              "success_rate": sum(r["is_success"] for r in results) / len(results),
              "episodes": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"success_rate={report['success_rate']:.1%}; report={args.output}")


if __name__ == "__main__":
    main()
