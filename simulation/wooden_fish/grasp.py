"""Contact-only gripping demonstration. No weld, attachment or external holding force."""
import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .scene import ROOT, build_tiger_model
from .env import WoodenFishEnv


def vector(a):
    return " ".join(map(str, a))


def build_grasp_model(tabletop, physics):
    root = ET.fromstring(build_tiger_model(tabletop, return_xml=True))
    grip = root.find(".//body[@name='gripper']")
    jaw = grip.find("body[@name='moving_jaw_so101_v1']")
    mallet = grip.find("body[@name='mallet']")
    grip.remove(mallet)
    root.find("worldbody").append(mallet)
    mallet.set("pos", "0 0 .3")
    ET.SubElement(mallet, "freejoint", name="mallet_free")
    # Remove every former jaw/mallet exclusion; ordinary contact is the only hold.
    contact = root.find("contact")
    for entry in list(contact):
        if entry.get("body1") == "mallet" or entry.get("body2") == "mallet":
            contact.remove(entry)
    for geom in mallet.findall("geom"):
        geom.set("friction", vector(physics["friction"]))
        geom.set("condim", "6")
        geom.set("solref", ".004 1")
        geom.set("solimp", ".99 .999 .0001")
        geom.set("mass", str(physics["handle_mass"] if geom.get("name") == "mallet_handle" else physics["head_mass"]))
    # Determine the moving pad's local frame at the mechanical closed limit.
    calibration = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    reference = mujoco.MjData(calibration)
    reference.qpos[5] = physics["close_angle"]
    mujoco.mj_forward(calibration, reference)
    bgrip = mujoco.mj_name2id(calibration, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    bjaw = mujoco.mj_name2id(calibration, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
    rg = reference.xmat[bgrip].reshape(3, 3)
    rj = reference.xmat[bjaw].reshape(3, 3)
    def jaw_pos(point):
        return rj.T @ (reference.xpos[bgrip] + rg @ np.asarray(point) - reference.xpos[bjaw])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, (rj.T @ rg).ravel())
    # Convex hulls of these concave finger meshes fill parts of the jaw gap.
    # Replace them with backing + pad solids, not collision-free fingers.
    for body in (grip, jaw):
        for geom in body.findall("geom"):
            if geom.get("class") == "collision" and geom.get("mesh") in (
                "wrist_roll_follower_so101_v1", "moving_jaw_so101_v1"
            ):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
    common = dict(type="box", friction=vector(physics["friction"]), condim="6",
                  solref=".004 1", solimp=".99 .999 .0001")
    ET.SubElement(grip, "geom", name="fixed_pad", pos="-.009 0 -.087",
                  size=vector(physics["pad_half_size"]), rgba=".04 .04 .04 1", **common)
    ET.SubElement(jaw, "geom", name="moving_pad", pos=vector(jaw_pos([-.002, 0, -.087])),
                  quat=vector(quat), size=vector(physics["pad_half_size"]), rgba=".04 .04 .04 1", **common)
    ET.SubElement(grip, "geom", name="fixed_finger_backing", pos="-.016 0 -.087",
                  size=".0055 .008 .015", rgba=".85 .85 .80 1", group="3", **common)
    ET.SubElement(jaw, "geom", name="moving_finger_backing", pos=vector(jaw_pos([.005, 0, -.087])),
                  quat=vector(quat), size=".0055 .008 .015", rgba=".85 .85 .80 1", group="3", **common)
    ET.SubElement(grip, "site", name="grasp_probe", pos="-.0054 0 -.068", size=".001", group="4")
    for actuator in root.find("actuator"):
        if actuator.get("name") == "gripper":
            actuator.set("kp", str(physics["gripper_kp"]))
            actuator.set("kv", str(physics["gripper_kv"]))
            limit = physics["gripper_torque_limit"]
            actuator.set("forcerange", f"{-limit} {limit}")
    root.find("option").set("timestep", str(physics["physics_timestep"]))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class ContactGrasp:
    def __init__(self, seed=0, physics=None, tabletop=None):
        self.tabletop = (tabletop if tabletop is not None
                         else json.loads((ROOT / "configs/tabletop-reference.json").read_text()))
        self.physics = physics or json.loads((ROOT / "configs/grasp-physics.json").read_text())
        self.model = build_grasp_model(self.tabletop, self.physics)
        self.data = mujoco.MjData(self.model)
        self.body = self.id("gripper", mujoco.mjtObj.mjOBJ_BODY)
        self.object_body = self.id("mallet", mujoco.mjtObj.mjOBJ_BODY)
        self.head = self.id("head", mujoco.mjtObj.mjOBJ_SITE)
        self.target = self.id("target", mujoco.mjtObj.mjOBJ_SITE)
        self.probe = self.id("grasp_probe", mujoco.mjtObj.mjOBJ_SITE)
        self.free = self.id("mallet_free", mujoco.mjtObj.mjOBJ_JOINT)
        self.qa = int(self.model.jnt_qposadr[self.free])
        self.va = int(self.model.jnt_dofadr[self.free])
        self.renderer = None
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [.20, 0, .10]
        self.camera.distance = .60
        self.camera.azimuth = 135
        self.camera.elevation = -25
        self.options = mujoco.MjvOption()
        self.options.geomgroup[3] = 0
        # Initialize only once. From here on, all object motion is integrated physics.
        baseline = WoodenFishEnv()
        try:
            baseline.reset(seed=seed)
            self.data.qpos[:6] = baseline.data.qpos[:6]
        finally:
            baseline.close()
        self.data.qpos[5] = self.physics["initial_angle"]
        self.data.ctrl[:] = self.data.qpos[:6]
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[self.qa:self.qa+3] = self.data.xpos[self.body] + self.rotation @ [-.0054, 0, -.068]
        self.data.qpos[self.qa+3:self.qa+7] = self.data.xquat[self.body]
        mujoco.mj_forward(self.model, self.data)
        self.original_head_z = float(self.data.site_xpos[self.head, 2])

    def id(self, name, kind=mujoco.mjtObj.mjOBJ_GEOM):
        value = mujoco.mj_name2id(self.model, kind, name)
        if value < 0:
            raise ValueError(f"Missing {name}")
        return value

    @property
    def rotation(self):
        return self.data.xmat[self.body].reshape(3, 3)

    def relative_object(self):
        return self.rotation.T @ (self.data.xpos[self.object_body] - self.data.xpos[self.body])

    def pad_forces(self):
        forces = {"fixed_pad": 0., "moving_pad": 0.}
        for index, contact in enumerate(self.data.contact):
            if self.id("mallet_handle") not in (contact.geom1, contact.geom2):
                continue
            for pad in forces:
                if self.id(pad) in (contact.geom1, contact.geom2):
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, index, wrench)
                    forces[pad] += float(wrench[0])
        return forces

    def tick(self, target=None, closed=True):
        if target is not None:
            self.data.ctrl[:5] = np.clip(target[:5], self.model.actuator_ctrlrange[:5, 0], self.model.actuator_ctrlrange[:5, 1])
        self.data.ctrl[5] = self.physics["close_angle"] if closed else self.physics["open_angle"]
        self.tap_impulse = 0.0
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            for index, contact in enumerate(self.data.contact):
                if {int(contact.geom1), int(contact.geom2)} == {self.id("mallet_head"), self.id("fish_target")}:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, index, force)
                    self.tap_impulse += max(0., force[0])*self.model.opt.timestep

    def ik_head(self, target):
        # This massless site is a kinematic probe, not an attachment or force.
        local = self.rotation.T @ (self.data.site_xpos[self.head] - self.data.xpos[self.body])
        self.model.site_pos[self.probe] = local
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        for _ in range(200):
            mujoco.mj_forward(self.model, scratch)
            error = np.asarray(target) - scratch.site_xpos[self.probe]
            if np.linalg.norm(error) < .0003:
                return scratch.qpos[:5].copy()
            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, scratch, jac, None, self.probe)
            j = jac[:, :5]
            dq = j.T @ np.linalg.solve(j @ j.T + np.eye(3)*1e-4, error)
            scratch.qpos[:5] = np.clip(scratch.qpos[:5] + np.clip(dq, -.05, .05),
                                      self.model.actuator_ctrlrange[:5, 0], self.model.actuator_ctrlrange[:5, 1])
        raise RuntimeError("Could not reach grasp-demo target")

    def render(self, camera="overview"):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.renderer.update_scene(self.data, camera=self.camera if camera == "overview" else camera, scene_option=self.options)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def rollout(sim):
    """Yield phases while applying only actuator targets; object qpos is never written."""
    for _ in range(40):
        sim.tick()
        yield "close"
    relative = sim.relative_object().copy()
    head_before = sim.data.site_xpos[sim.head].copy()
    goal = sim.ik_head(head_before + [0, 0, .04])
    start = sim.data.ctrl[:5].copy()
    for i in range(60):
        sim.tick(start + (goal-start)*min(1, (i+1)/50))
        yield "lift"
    max_drift = 0.
    min_forces = {"fixed_pad": float("inf"), "moving_pad": float("inf")}
    for _ in range(100):
        sim.tick()
        max_drift = max(max_drift, float(np.linalg.norm(sim.relative_object()-relative)))
        for pad, force in sim.pad_forces().items():
            min_forces[pad] = min(min_forces[pad], force)
        yield "hold"
    held_height = float(sim.data.site_xpos[sim.head, 2])
    report = {"head_lift_m": held_height-float(head_before[2]), "max_relative_drift_m": max_drift,
              "min_pad_normal_force_N": min_forces,
              "held_gripper_angle_rad": float(sim.data.qpos[5]),
              "no_attachment": sim.model.body_parentid[sim.object_body] == 0 and sim.model.neq == 0}
    hit = False
    impulse = 0.
    for _ in range(150):
        gid = sim.id("mallet_head")
        vertical = sim.data.geom_xmat[gid].reshape(3, 3)[2]
        radius = float(np.linalg.norm(vertical*sim.model.geom_size[gid]))
        target = sim.data.site_xpos[sim.target] + [0, 0, radius-.001]
        desired = sim.ik_head(target)
        sim.tick(sim.data.ctrl[:5] + np.clip(desired-sim.data.ctrl[:5], -.012, .012))
        impulse += sim.tap_impulse
        yield "tap"
        if impulse > 1e-5:
            hit = True
            break
    lift_target = sim.data.site_xpos[sim.head].copy() + [0, 0, .045]
    for _ in range(70):
        desired = sim.ik_head(lift_target)
        sim.tick(sim.data.ctrl[:5] + np.clip(desired-sim.data.ctrl[:5], -.01, .01))
        yield "retract"
    report["tap_contact"] = hit
    report["tap_impulse_Ns"] = impulse
    report["post_tap_pad_force_N"] = sim.pad_forces()
    report["post_tap_relative_drift_m"] = float(np.linalg.norm(sim.relative_object()-relative))
    held_height = float(sim.data.site_xpos[sim.head, 2])
    for _ in range(150):
        sim.tick(closed=False)
        yield "release"
    report["release_drop_m"] = held_height-float(sim.data.site_xpos[sim.head, 2])
    report["passed"] = bool(report["no_attachment"] and report["head_lift_m"] > .025 and
                            max_drift < .005 and min(min_forces.values()) > .01 and report["release_drop_m"] > .05
                            and hit and min(report["post_tap_pad_force_N"].values()) > .01
                            and report["post_tap_relative_drift_m"] < .005)
    yield report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--camera", choices=["overview", "side", "wrist"], default="overview")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"runs/contact-grasp.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.speed <= 0:
        parser.error("speed must be positive")
    sim = ContactGrasp(seed=args.seed)
    viewer = None
    writer = None
    try:
        if args.live:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
            with viewer.lock():
                viewer.cam.lookat[:] = sim.camera.lookat
                viewer.cam.distance = sim.camera.distance
                viewer.cam.azimuth = sim.camera.azimuth
                viewer.cam.elevation = sim.camera.elevation
                viewer.opt.geomgroup[3] = 0
                if args.camera != "overview":
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = sim.id(args.camera, mujoco.mjtObj.mjOBJ_CAMERA)
        if args.video:
            import imageio.v2 as imageio
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(args.video, fps=50)
        from contextlib import nullcontext
        iterator = rollout(sim)
        last = None
        while True:
            if viewer is not None and not viewer.is_running():
                print("Closed by user; validation incomplete.")
                break
            start = time.monotonic()
            with viewer.lock() if viewer is not None else nullcontext():
                phase = next(iterator)
                if writer and not isinstance(phase, dict):
                    writer.append_data(sim.render(args.camera))
            if isinstance(phase, dict):
                args.output.parent.mkdir(parents=True, exist_ok=True)
                phase["physics"] = sim.physics
                args.output.write_text(json.dumps(phase, indent=2)+"\n")
                print(json.dumps(phase, indent=2))
                if not phase["passed"]:
                    raise RuntimeError("Physical grasp validation failed")
                break
            if phase != last:
                print(f"Phase: {phase}", flush=True)
                last = phase
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0,.02/args.speed-(time.monotonic()-start)))
    finally:
        if viewer is not None:
            viewer.close()
        if writer:
            writer.close()
        sim.close()


if __name__ == "__main__":
    main()
