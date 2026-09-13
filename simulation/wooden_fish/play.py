"""Live scripted tapping demo. macOS: mjpython -m wooden_fish.play."""
import argparse
import time
import mujoco.viewer
from .env import WoodenFishEnv
from .run import scripted_action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1, help="Contact demo runs once; fixed-grip mode supports 0 to repeat")
    parser.add_argument("--speed", type=float, default=0.5, help="Playback speed, default half speed")
    parser.add_argument("--camera", choices=["overview", "side", "wrist"], default="overview")
    parser.add_argument("--fixed-grip", action="store_true", help="Use the historical attached-mallet demonstration")
    args = parser.parse_args()
    if args.speed <= 0 or args.episodes < 0:
        parser.error("speed must be positive and episodes nonnegative")
    if not args.fixed_grip:
        if args.episodes != 1:
            parser.error("Contact grasp validation runs one episode; use --episodes 1")
        from .grasp import main as grasp_main
        grasp_main(["--live", "--camera", args.camera, "--speed", str(args.speed)])
        return
    env = WoodenFishEnv()
    env.reset(seed=10000)
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = env.camera.lookat
                viewer.cam.distance = env.camera.distance
                viewer.cam.azimuth = env.camera.azimuth
                viewer.cam.elevation = env.camera.elevation
                viewer.opt.geomgroup[3] = 0
                if args.camera != "overview":
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
            print("Tiger scene: scripted single tap, starting with mallet held; half-speed by default.", flush=True)
            episode = 0
            while viewer.is_running() and (args.episodes == 0 or episode < args.episodes):
                with viewer.lock():
                    env.reset(seed=10000+episode)
                until = time.monotonic()+1
                while viewer.is_running() and time.monotonic()<until:
                    viewer.sync(); time.sleep(.02)
                if not viewer.is_running():
                    break
                while viewer.is_running():
                    start = time.monotonic()
                    with viewer.lock():
                        _, _, done, timeout, info = env.step(scripted_action(env))
                    viewer.sync()
                    time.sleep(max(0, .02/args.speed-(time.monotonic()-start)))
                    if done or timeout:
                        print(f"Episode {episode+1}: {info['reason']}, hits={info['hits']}", flush=True)
                        break
                until = time.monotonic()+1
                while viewer.is_running() and time.monotonic()<until:
                    viewer.sync(); time.sleep(.02)
                episode += 1
    finally:
        env.close()


if __name__ == "__main__":
    main()
