"""Photo-informed tabletop layout preview, separate from the validated RL baseline.

All bodies are fixed in this review scene. This is not a grasping environment,
acoustic model, metric reconstruction or calibrated collision model.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import imageio.v2 as imageio
import mujoco
import numpy as np
from .scene import ROOT


def v(values):
    return " ".join(f"{x:.8g}" for x in values)


def geom(parent, name, kind, pos, size, colour, *, visual=False, **attrs):
    return ET.SubElement(parent, "geom", name=name, type=kind, pos=v(pos), size=v(size),
                         rgba=v(colour), contype="0" if visual else "1",
                         conaffinity="0" if visual else "1", **attrs)


def build_tabletop(config):
    p, f, m, mat = [config[key] for key in ("plate", "fish", "mallet", "mat")]
    for group, keys in ((p, ("length", "width", "thickness")),
                        (f, ("width", "depth", "height")),
                        (m, ("handle_length", "head_radius", "head_length", "handle_diameter")),
                        (mat, ("length", "width", "grid_spacing"))):
        if any(not np.isfinite(group[k]) or group[k] <= 0 for k in keys):
            raise ValueError("All dimensions must be finite and positive")
    if m["handle_length"] <= m["handle_diameter"]:
        raise ValueError("Handle length must exceed its diameter")
    root = ET.Element("mujoco", model="photo_reference_tabletop")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1000", offheight="800")
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.7 0.7 0.7")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -0.15 0.65", dir="0 0 -1")
    geom(world, "table", "plane", (0, 0, -0.003), (0.6, 0.6, 0.01), (0.63, 0.60, 0.52, 1))
    geom(world, "cutting_mat", "box", (0, 0, -0.0015),
         (mat["length"]/2, mat["width"]/2, 0.0015), (0.015, 0.30, 0.27, 1))
    # Vector geometry grid: no generated or edited photo texture.
    for axis, extent, other in ((0, mat["length"], mat["width"]), (1, mat["width"], mat["length"])):
        for i, coordinate in enumerate(np.arange(-extent/2, extent/2 + 1e-8, mat["grid_spacing"])):
            pos = [0, 0, 0.00003]; pos[axis] = coordinate
            size = [other/2, other/2, 0.00002]; size[axis] = 0.00009
            geom(world, f"grid_{axis}_{i}", "box", pos, size, (0.55, 0.79, 0.73, 1), visual=True)
    plate = ET.SubElement(world, "body", name="support_plate")
    geom(plate, "plate", "box", (0, 0, p["thickness"]/2),
         (p["length"]/2, p["width"]/2, p["thickness"]/2), (0.70, 0.72, 0.73, 1))
    z = p["thickness"]
    for i, y in enumerate((-p["width"]*.34, p["width"]*.34)):
        geom(plate, f"round_pad_{i}", "cylinder", (p.get("round_pad_side", -1)*(p["length"]/2 - 0.012), y, z+0.0003),
             (0.007, 0.0003), (0.36, 0.37, 0.38, 1), visual=True)
    geom(plate, "long_pad", "box", (-p.get("round_pad_side", -1)*(p["length"]/2-0.009), 0, z+0.0003),
         (0.0025, p["width"]*0.38, 0.0003), (0.34, 0.35, 0.35, 1), visual=True)
    # Ellipsoids approximate the rounded tiger shell and its dark mouth slit.
    w, d, h = f["width"], f["depth"], f["height"]
    fish = ET.SubElement(world, "body", name="tiger_fish", pos=v((*f["centre_xy"], z)))
    orange = (1.0, 0.38, 0.015, 1)
    geom(fish, "fish_lower", "ellipsoid", (0, 0, h*.20), (w*.48, d*.48, h*.20), (0.84, 0.26, 0.015, 1))
    geom(fish, "mouth_slit", "ellipsoid", (0, -d*.025, h*.34), (w*.475, d*.48, h*.055), (0.25, 0.035, 0.005, 1), visual=True)
    geom(fish, "fish_shell", "ellipsoid", (0, 0, h*.565), (w*.5, d*.47, h*.355), orange)
    for i, x in enumerate((-w*.29, w*.29)):
        geom(fish, f"ear_{i}", "ellipsoid", (x, d*.11, h*.83), (w*.20, d*.18, h*.17), orange)
        geom(fish, f"ear_pink_{i}", "ellipsoid", (x, -d*.08, h*.855), (w*.12, d*.032, h*.078), (1, .66, .65, 1), visual=True)
    # Surface details placed approximately on the curved front, entirely visual.
    geom(fish, "muzzle", "ellipsoid", (0, -d*.426, h*.46), (w*.28, d*.044, h*.115), (1, .84, .76, 1), visual=True)
    for i, x in enumerate((-w*.17, w*.17)):
        geom(fish, f"eye_{i}", "ellipsoid", (x, -d*.448, h*.56), (w*.045, d*.024, h*.045), (.025,.015,.01,1), visual=True)
        geom(fish, f"eye_glint_{i}", "sphere", (x-w*.01, -d*.471, h*.575), (w*.009,), (1,1,1,1), visual=True)
    geom(fish, "nose", "ellipsoid", (0, -d*.474, h*.535), (w*.037, d*.02, h*.029), (.03,.018,.01,1), visual=True)
    for side in (-1,1):
        for j in range(2):
            geom(fish, f"stripe_{side}_{j}", "ellipsoid", (side*w*.34,-d*.327,h*(.52+j*.085)),
                 (w*.045,d*.022,h*.017), (.5,.15,.025,1), visual=True)
    # A small forehead motif; shape approximation, not a scanned texture.
    for i in range(3):
        zz = h*(.70 + i*.045)
        yy = -d*.47 * np.sqrt(max(0,1-((zz-h*.565)/(h*.355))**2))
        geom(fish, f"forehead_{i}", "box", (0,yy-.0002,zz), (w*.065,.0005,h*.009), (.58,.18,.025,1), visual=True)
    # Marker only: not yet used as the validated RL contact target.
    ET.SubElement(fish, "site", name="candidate_strike_point", pos=v((0,-d*.09,h*.906)), size="0.0015", rgba="0 0.8 0.5 0.7", group="4")
    # Resting mallet: white oval head on the plate, wooden handle projects past it.
    radius = m["head_radius"]
    mallet = ET.SubElement(world, "body", name="resting_mallet", pos=v((*m["head_xy"], z+radius)))
    geom(mallet, "resting_head", "ellipsoid", (0,0,0), (radius,m["head_length"]/2,radius), (.93,.89,.80,1))
    direction = m.get("handle_direction_y", 1)
    if direction not in (-1, 1):
        raise ValueError("handle_direction_y must be -1 or 1")
    handle_start = m["head_length"]*.3
    handle_end = handle_start + m["handle_length"] - m["handle_diameter"]
    ET.SubElement(mallet, "geom", name="resting_handle", type="capsule",
                  fromto=v((0,direction*handle_start,0,0,direction*handle_end,0)), size=str(m["handle_diameter"]/2), rgba="0.60 0.34 0.15 1")
    return ET.tostring(root,encoding="unicode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT/"configs/tabletop-reference.json")
    parser.add_argument("--output", type=Path, default=ROOT/"runs/tabletop-reference")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    xml = build_tabletop(config)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"scene.xml").write_text(xml)
    (args.output/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0,0.005,0.025]
    camera.distance = .34
    with mujoco.Renderer(model,height=720,width=960) as renderer:
        for name, azimuth, elevation in (("perspective",65,-30),("top",90,-89),("front",90,-8)):
            camera.azimuth = azimuth; camera.elevation=elevation
            renderer.update_scene(data,camera=camera)
            imageio.imwrite(args.output/f"{name}.png",renderer.render())
    print(json.dumps({"output":str(args.output),"status":config["status"],"bodies":model.nbody,
                      "note":"User dimensions applied; shape and placement partly estimated. Fixed preview, not a calibrated RL scene."}))


if __name__ == "__main__":
    main()
