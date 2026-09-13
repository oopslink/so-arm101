import numpy as np
import mujoco
import pytest
from stable_baselines3.common.env_checker import check_env
from wooden_fish.env import WoodenFishEnv, TaskConfig
from wooden_fish.hit_detector import HitDetector
from wooden_fish.run import scripted_action


def test_hit_requires_valid_impact_and_release():
    detector = HitDetector()
    assert detector.update(True, False, 0.1, 0)[0] is False
    detector.update(False, True, 0.1, 0.05)
    assert detector.update(True, True, 0.1, 0)[0] is True
    for _ in range(100):
        assert detector.update(True, True, 0.1, 0) == (False, False)
    assert detector.count == 1 and not detector.success
    for _ in range(19):
        detector.update(False, True, 0, 0.03)
        assert not detector.success
    detector.update(False, True, 0, 0.03)
    assert detector.success


@pytest.mark.parametrize("speed", [0, -0.1, 1.0])
def test_invalid_speed_does_not_count(speed):
    detector = HitDetector()
    detector.update(True, True, speed, 0)
    assert detector.count == 0


def test_extra_impact_is_reported():
    detector = HitDetector()
    detector.update(True, True, 0.1, 0)
    detector.update(False, True, 0, 0.01)
    assert detector.update(True, True, 0.1, 0) == (False, True)


def test_contract_seed_and_timeout():
    env = WoodenFishEnv(config=TaskConfig(max_steps=2))
    try:
        check_env(env)
        a, _ = env.reset(seed=42)
        b, _ = env.reset(seed=42)
        np.testing.assert_array_equal(a, b)
        _, _, done, timeout, info = env.step(np.zeros(5))
        assert not done and not timeout and info["hits"] == 0
        _, _, done, timeout, info = env.step(np.zeros(5))
        assert not done and timeout and not info["is_success"]
        with pytest.raises(RuntimeError):
            env.step(np.zeros(5))
        env.reset()
        with pytest.raises(ValueError):
            env.step([np.nan] * 5)
    finally:
        env.close()


@pytest.mark.parametrize("seed", [100, 101, 102])
def test_scripted_physical_tap(seed):
    env = WoodenFishEnv()
    try:
        env.reset(seed=seed)
        assert env.data.ncon == 0
        for _ in range(100):
            _, _, done, timeout, info = env.step(scripted_action(env))
            if done or timeout:
                break
        assert info["is_success"] and info["hits"] == 1, info
    finally:
        env.close()


def test_table_collision_is_failure():
    env = WoodenFishEnv()
    try:
        env.reset(seed=0)
        # Move the whole static table body up to the starting head.
        table = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "tabletop")
        env.model.body_pos[table, 2] = env.data.site_xpos[env.head, 2]
        mujoco.mj_forward(env.model, env.data)
        _, _, done, timeout, info = env.step(np.zeros(5))
        assert done and not timeout
        assert info["reason"] == "collision" and not info["is_success"]
    finally:
        env.close()


def test_saved_legacy_config_keeps_cylinder():
    env = WoodenFishEnv(config=TaskConfig.from_saved({}))
    try:
        assert env.config.scene_profile == "cylinder"
        env.reset(seed=0)
        for _ in range(100):
            _, _, done, timeout, info = env.step(scripted_action(env))
            if done or timeout:
                break
        assert info["is_success"]
    finally:
        env.close()


def test_tiger_dimensions_and_no_duplicate_mallet():
    env = WoodenFishEnv()
    try:
        assert env.config.scene_profile == "tiger"
        geom = lambda name: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        np.testing.assert_allclose(2*env.model.geom_size[geom("plate")], [.13,.07,.008])
        np.testing.assert_allclose(env.model.geom_size[geom("mallet_head")], [.006,.006,.008])
        r,h,_ = env.model.geom_size[geom("mallet_handle")]
        np.testing.assert_allclose(2*h+2*r,.10)
        assert geom("resting_head") == -1
    finally:
        env.close()


def test_side_camera_fixed_and_wrist_camera_moves():
    env = WoodenFishEnv()
    try:
        env.reset(seed=0)
        side = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "side")
        wrist = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        assert side >= 0 and wrist >= 0
        side_start = env.data.cam_xpos[side].copy()
        wrist_start = env.data.cam_xpos[wrist].copy()
        env.step(scripted_action(env))
        np.testing.assert_allclose(env.data.cam_xpos[side], side_start)
        assert np.linalg.norm(env.data.cam_xpos[wrist]-wrist_start) > 1e-6
    finally:
        env.close()
