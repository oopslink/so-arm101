"""Versioned MuJoCo scene registry for SO-101 Lab.

The registry keeps task intent, assets, cameras and validation limits together.
It does not claim that a partial physics scene is a calibrated digital twin.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco

from .grasp import ContactGrasp, build_grasp_model
from .scene import ROOT


SCENES_DIR = ROOT / "configs" / "scenes"
REQUIRED_FIELDS = {
    "schema_version",
    "id",
    "version",
    "title",
    "description",
    "engine",
    "kind",
    "status",
    "assets",
    "robot",
    "cameras",
    "task",
    "success_criteria",
}
SUPPORTED_KINDS = {"contact_grasp"}


def _inside_root(relative_path: str) -> Path:
    path = (ROOT / relative_path).resolve()
    root = ROOT.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Scene asset escapes simulation root: {relative_path}")
    if not path.is_file():
        raise ValueError(f"Scene asset does not exist: {relative_path}")
    return path


@dataclass(frozen=True)
class SceneSpec:
    path: Path
    values: dict[str, Any]

    @property
    def scene_id(self) -> str:
        return str(self.values["id"])

    def asset_path(self, name: str) -> Path:
        try:
            relative_path = self.values["assets"][name]
        except KeyError as error:
            raise ValueError(f"Scene {self.scene_id} is missing asset {name}") from error
        return _inside_root(relative_path)

    def read_asset_json(self, name: str) -> dict[str, Any]:
        return json.loads(self.asset_path(name).read_text())

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.scene_id,
            "version": self.values["version"],
            "title": self.values["title"],
            "engine": self.values["engine"],
            "kind": self.values["kind"],
            "status": self.values["status"],
        }


def scene_files() -> list[Path]:
    return sorted(SCENES_DIR.glob("*.json"))


def load_scene(scene_id: str) -> SceneSpec:
    matches = []
    for path in scene_files():
        values = json.loads(path.read_text())
        if values.get("id") == scene_id:
            matches.append(SceneSpec(path, values))
    if not matches:
        available = ", ".join(path.stem for path in scene_files()) or "none"
        raise ValueError(f"Unknown scene {scene_id!r}; available manifests: {available}")
    if len(matches) > 1:
        raise ValueError(f"Duplicate scene id: {scene_id}")
    return matches[0]


def list_scenes() -> list[SceneSpec]:
    specs = []
    ids = set()
    for path in scene_files():
        values = json.loads(path.read_text())
        scene_id = values.get("id")
        if not scene_id or scene_id in ids:
            raise ValueError(f"Missing or duplicate scene id in {path}")
        ids.add(scene_id)
        specs.append(SceneSpec(path, values))
    return specs


def validate_spec(spec: SceneSpec, *, compile_model: bool = True) -> dict[str, Any]:
    values = spec.values
    missing = sorted(REQUIRED_FIELDS - values.keys())
    if missing:
        raise ValueError(f"Scene {spec.scene_id} is missing fields: {', '.join(missing)}")
    if values["schema_version"] != 1:
        raise ValueError(f"Unsupported scene schema: {values['schema_version']}")
    if values["engine"] != "mujoco":
        raise ValueError("Only the MuJoCo engine is supported")
    if values["kind"] not in SUPPORTED_KINDS:
        raise ValueError(f"Unsupported scene kind: {values['kind']}")
    action_order = values["robot"].get("action_order", [])
    if len(action_order) != 6 or len(set(action_order)) != 6:
        raise ValueError("SO-101 action_order must contain six unique controls")
    cameras = values["cameras"]
    if len(cameras) != len(set(cameras)) or not {"side", "wrist"}.issubset(cameras):
        raise ValueError("Scene must define unique side and wrist cameras")
    tabletop = spec.read_asset_json("tabletop_config")
    physics = spec.read_asset_json("physics_config")
    spec.asset_path("robot_model")
    result = {**spec.summary(), "valid": True, "compiled": False}
    if compile_model:
        model = build_grasp_model(tabletop, physics)
        names = {
            "side": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side"),
            "wrist": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist"),
            "mallet": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mallet"),
            "fish": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fish"),
        }
        missing_names = sorted(name for name, identifier in names.items() if identifier < 0)
        if missing_names:
            raise ValueError(f"Compiled model is missing: {', '.join(missing_names)}")
        if model.nu != len(action_order):
            raise ValueError(f"Action contract has {len(action_order)} values but model exposes {model.nu}")
        actuator_order = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in range(model.nu)
        ]
        if actuator_order != action_order:
            raise ValueError(f"Action order {action_order} does not match actuators {actuator_order}")
        result.update(
            compiled=True,
            model={
                "nq": model.nq,
                "nv": model.nv,
                "nu": model.nu,
                "ncam": model.ncam,
                "actuators": actuator_order,
            },
        )
    return result


def build_scene(spec: SceneSpec) -> mujoco.MjModel:
    validate_spec(spec, compile_model=False)
    return build_grasp_model(
        spec.read_asset_json("tabletop_config"),
        spec.read_asset_json("physics_config"),
    )


def export_scene(spec: SceneSpec, output: Path) -> dict[str, Any]:
    model = build_scene(spec)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(output), model)
    metadata = output.with_suffix(".scene.json")
    metadata.write_text(json.dumps(spec.values, ensure_ascii=False, indent=2) + "\n")
    return {"scene": spec.scene_id, "mjcf": str(output), "metadata": str(metadata)}


def render_scene(spec: SceneSpec, output: Path, camera: str) -> dict[str, Any]:
    validate_spec(spec, compile_model=False)
    if camera not in spec.values["cameras"]:
        raise ValueError(f"Camera {camera!r} is not declared by scene {spec.scene_id}")
    simulator = ContactGrasp(
        tabletop=spec.read_asset_json("tabletop_config"),
        physics=spec.read_asset_json("physics_config"),
    )
    try:
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(output, simulator.render(camera))
    finally:
        simulator.close()
    return {"scene": spec.scene_id, "camera": camera, "image": str(output)}


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manage versioned SO-101 MuJoCo scenes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List registered scenes")
    show = subparsers.add_parser("show", help="Show one scene manifest")
    show.add_argument("scene_id")
    validate = subparsers.add_parser("validate", help="Validate and compile scenes")
    validate.add_argument("scene_id", nargs="?")
    export = subparsers.add_parser("export", help="Export compiled MJCF and scene metadata")
    export.add_argument("scene_id")
    export.add_argument("--output", type=Path, required=True)
    render = subparsers.add_parser("render", help="Render an initialized scene camera")
    render.add_argument("scene_id")
    render.add_argument("--camera", choices=("overview", "side", "wrist"), default="overview")
    render.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "list":
        _print_json([spec.summary() for spec in list_scenes()])
    elif args.command == "show":
        _print_json(load_scene(args.scene_id).values)
    elif args.command == "validate":
        specs = [load_scene(args.scene_id)] if args.scene_id else list_scenes()
        _print_json([validate_spec(spec) for spec in specs])
    elif args.command == "export":
        _print_json(export_scene(load_scene(args.scene_id), args.output))
    elif args.command == "render":
        _print_json(render_scene(load_scene(args.scene_id), args.output, args.camera))


if __name__ == "__main__":
    main()
