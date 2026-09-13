"""Interactive tabletop viewer. On macOS run this module with mjpython."""
import argparse
from pathlib import Path
import time
import mujoco
import mujoco.viewer
from .scene import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=ROOT / "runs/tabletop-reference/scene.xml")
    parser.add_argument("--seconds", type=float, help="Close automatically after this many seconds")
    args = parser.parse_args()
    scene = args.scene.expanduser().resolve()
    if not scene.is_file():
        parser.error(f"Scene does not exist: {scene}. Run wooden_fish.tabletop first.")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0, 0.005, 0.025]
            viewer.cam.distance = 0.34
            viewer.cam.azimuth = 65
            viewer.cam.elevation = -30
        print(f"Viewer opened: {scene}", flush=True)
        start = time.monotonic()
        while viewer.is_running():
            if args.seconds is not None and time.monotonic() - start >= args.seconds:
                break
            # Fixed-body layout inspection, not a physics/control rollout.
            viewer.sync()
            time.sleep(1 / 60)
    print("Viewer closed.", flush=True)


if __name__ == "__main__":
    main()
