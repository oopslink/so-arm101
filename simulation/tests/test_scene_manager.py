import json

import mujoco

from wooden_fish.scene_manager import (
    build_scene,
    list_scenes,
    load_scene,
    main,
    validate_spec,
)


SCENE_ID = "wooden-fish-contact-v1"


def test_registry_loads_versioned_scene():
    scenes = list_scenes()
    assert [scene.scene_id for scene in scenes] == [SCENE_ID]
    scene = load_scene(SCENE_ID)
    assert scene.values["engine"] == "mujoco"
    assert scene.values["task"]["not_implemented_phases"]


def test_scene_contract_matches_compiled_mujoco_model():
    scene = load_scene(SCENE_ID)
    result = validate_spec(scene)
    assert result["valid"] and result["compiled"]
    assert result["model"]["nu"] == 6
    assert result["model"]["actuators"] == scene.values["robot"]["action_order"]
    model = build_scene(scene)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist") >= 0


def test_cli_list_is_machine_readable(capsys):
    main(["list"])
    values = json.loads(capsys.readouterr().out)
    assert values[0]["id"] == SCENE_ID
