from dataclasses import dataclass
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from .scene import build_model, build_tiger_model, HEAD_RADIUS, ROOT
from .hit_detector import HitDetector


@dataclass
class TaskConfig:
    scene_profile: str = "tiger"
    tabletop_config: dict | None = None
    frame_skip: int = 10
    max_steps: int = 250
    action_scale: float = 0.015  # radians per 20 ms control step
    target_jitter: float = 0.005  # metres in x/y
    start_jitter: float = 0.003  # radians


    def __post_init__(self):
        if self.scene_profile not in ("tiger", "cylinder"):
            raise ValueError("scene_profile must be tiger or cylinder")
        if self.scene_profile == "tiger" and self.tabletop_config is None:
            self.tabletop_config = json.loads((ROOT / "configs/tabletop-reference.json").read_text())

    @classmethod
    def from_saved(cls, values):
        values = dict(values)
        values.setdefault("scene_profile", "cylinder")
        return cls(**values)


class WoodenFishEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, config=None):
        self.config = config or TaskConfig()
        if self.config.frame_skip != 10 or self.config.max_steps < 1 or self.config.action_scale <= 0:
            raise ValueError("Baseline uses 500 Hz physics / 50 Hz control and positive limits")
        if self.config.target_jitter < 0 or self.config.start_jitter < 0:
            raise ValueError("Jitter must be nonnegative")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Only rgb_array rendering is supported")
        self.render_mode = render_mode
        self.model = (build_tiger_model(self.config.tabletop_config)
                      if self.config.scene_profile == "tiger" else build_model())
        self.data = mujoco.MjData(self.model)
        self.head = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "head")
        self.target_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        self.fish_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "fish")
        self.geom = {name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                     for name in ("mallet_head", "mallet_handle", "table", "fish_target", "fish_base")}
        self.fish_origin = self.model.body_pos[self.fish_body, :2].copy()
        self.hit_region = 0.012 if self.config.scene_profile == "tiger" else 0.035
        self.low = self.model.actuator_ctrlrange[:5, 0].copy()
        self.high = self.model.actuator_ctrlrange[:5, 1].copy()
        self.action_space = spaces.Box(-1.0, 1.0, (5,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (32,), dtype=np.float32)
        self.renderer = None
        self.scene_option = mujoco.MjvOption()
        self.scene_option.geomgroup[3] = 0  # Hide collision meshes in the video.
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.15, 0, 0.15]
        self.camera.distance = 0.85
        self.camera.azimuth = 135
        self.camera.elevation = -25
        if self.config.scene_profile == "tiger":
            self.camera.distance = 0.65
            self.camera.lookat[:] = [0.19, 0, 0.10]
        self._done = True

    @property
    def target(self):
        return self.data.site_xpos[self.target_site].copy()

    @property
    def head_radius(self):
        if self.config.scene_profile == "cylinder":
            return HEAD_RADIUS
        gid = self.geom["mallet_head"]
        vertical = self.data.geom_xmat[gid].reshape(3, 3)[2]
        return float(np.linalg.norm(vertical * self.model.geom_size[gid]))

    def head_velocity(self):
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac, None, self.head)
        return jac @ self.data.qvel

    def solve_ik(self, position, seed=None):
        """Position-only damped IK for setup and scripted validation, not the policy."""
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = seed if seed is not None else [0, -0.5, 0.5, 0.5, 0, 0.4]
        for _ in range(250):
            mujoco.mj_forward(self.model, scratch)
            error = np.asarray(position) - scratch.site_xpos[self.head]
            if np.linalg.norm(error) < 0.0002:
                return scratch.qpos.copy()
            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, scratch, jac, None, self.head)
            j = jac[:, :5]
            dq = j.T @ np.linalg.solve(j @ j.T + np.eye(3) * 1e-4, error)
            scratch.qpos[:5] = np.clip(scratch.qpos[:5] + np.clip(dq, -0.1, 0.1), self.low, self.high)
        raise ValueError(f"IK failed for {position}: residual {np.linalg.norm(error):.4f} m")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_pos[self.fish_body, :2] = self.fish_origin + self.np_random.uniform(
            -self.config.target_jitter, self.config.target_jitter, 2)
        mujoco.mj_forward(self.model, self.data)
        start = self.target + [0, 0, self.head_radius + 0.06]
        self.data.qpos[:] = self.solve_ik(start)
        self.data.qpos[:5] = np.clip(self.data.qpos[:5] + self.np_random.uniform(
            -self.config.start_jitter, self.config.start_jitter, 5), self.low, self.high)
        self.data.ctrl[:] = self.data.qpos
        # Let servo dynamics settle before the episode. Reset time/velocity afterwards.
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.detector = HitDetector()
        self.steps = 0
        self.last_action = np.zeros(5)
        self._done = False
        return self._obs(), self._info("running")

    def _obs(self):
        return np.concatenate((
            self.data.qpos, self.data.qvel, self.data.ctrl[:5],
            self.data.site_xpos[self.head] - self.target, self.head_velocity(),
            self.last_action, [self.detector.count, self.detector.released / 20,
                               self.steps / self.config.max_steps, float(self.detector.touching)],
        )).astype(np.float32)

    def _info(self, reason):
        return {"is_success": bool(self.detector.success and reason == "success"),
                "scene_profile": self.config.scene_profile, "hits": self.detector.count, "reason": reason, "steps": self.steps,
                "head_position": self.data.site_xpos[self.head].tolist(),
                "target_position": self.target.tolist()}

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset() before stepping a completed episode")
        action = np.asarray(action, dtype=float)
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError("Expected five finite action values")
        action = np.clip(action, -1, 1)
        before = self.data.site_xpos[self.head].copy()
        was_hit = bool(self.detector.count)
        goal = self.target + [0, 0, self.head_radius + (0.045 if was_hit else 0)]
        old_distance = np.linalg.norm(before - goal)
        self.data.ctrl[:5] = np.clip(self.data.ctrl[:5] + action * self.config.action_scale, self.low, self.high)
        reward = -0.002 - 0.005 * float(np.square(action - self.last_action).mean())
        reason = "running"
        for _ in range(self.config.frame_skip):
            downward_speed = -float(self.head_velocity()[2])
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            head = self.data.site_xpos[self.head]
            target_contact = False
            bad_contact = False
            for c in self.data.contact:
                pair = {int(c.geom1), int(c.geom2)}
                if pair == {self.geom["mallet_head"], self.geom["fish_target"]}:
                    target_contact = True
                else:
                    # Any other active collision invalidates this contact-task baseline.
                    bad_contact = True
            in_region = np.linalg.norm(head[:2] - self.target[:2]) <= self.hit_region
            gap = head[2] - self.head_radius - self.target[2]
            hit, repeated = self.detector.update(target_contact, in_region, downward_speed, gap)
            reward += 3.0 * hit
            if bad_contact or repeated:
                reason = "collision" if bad_contact else "extra_hit"
                reward -= 5
                break
            if self.detector.success:
                reason = "success"
                reward += 10
                break
        self.steps += 1
        reward += 10 * (old_distance - np.linalg.norm(self.data.site_xpos[self.head] - goal))
        self.last_action = action.copy()
        terminated = reason != "running"
        truncated = not terminated and self.steps >= self.config.max_steps
        if truncated:
            reason = "timeout"
        self._done = terminated or truncated
        return self._obs(), float(reward), terminated, truncated, self._info(reason)

    def render(self, camera=None):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.renderer.update_scene(self.data, camera=self.camera if camera is None else camera, scene_option=self.scene_option)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
