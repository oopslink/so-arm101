"""Compose a task around the unmodified upstream MJCF."""
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco

ROOT = Path(__file__).resolve().parents[1]
HEAD_RADIUS = 0.016
FISH_TOP = 0.105


def build_model():
    source = ROOT / "assets/so101"
    tree = ET.parse(source / "so101_new_calib.xml")
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(source / "assets"))
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="720")
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.7 0.7 0.7")
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="0 -0.2 1.5", dir="0 0 -1")
    ET.SubElement(world, "geom", name="table", type="plane", size="0.6 0.6 0.02", rgba="0.18 0.22 0.27 1")
    fish = ET.SubElement(world, "body", name="fish", pos="0.25 0 0")
    ET.SubElement(fish, "geom", name="fish_base", type="cylinder", pos="0 0 0.045", size="0.052 0.045", rgba="0.75 0.26 0.06 1")
    ET.SubElement(fish, "geom", name="fish_target", type="cylinder", pos="0 0 0.0975", size="0.04 0.0075", friction="0.7 0.005 0.0001", rgba="0.95 0.48 0.12 1")
    ET.SubElement(fish, "site", name="target", pos=f"0 0 {FISH_TOP}", size="0.004", rgba="0.1 0.9 0.5 1")
    gripper = root.find(".//body[@name='gripper']")
    mallet = ET.SubElement(gripper, "body", name="mallet", pos="-0.0079 -0.000218121 -0.0981274")
    ET.SubElement(mallet, "geom", name="mallet_handle", type="capsule", fromto="0 0 0 0 0 -0.10", size="0.006", mass="0.018", rgba="0.65 0.4 0.2 1")
    ET.SubElement(mallet, "geom", name="mallet_head", type="sphere", pos="0 0 -0.115", size=str(HEAD_RADIUS), mass="0.025", rgba="0.45 0.22 0.10 1")
    ET.SubElement(mallet, "site", name="head", pos="0 0 -0.115", size="0.003", rgba="1 0 0 1")
    contact = ET.SubElement(root, "contact")
    # The first task uses a rigid fixture, so adjacent jaw/fixture contact is excluded.
    for body in ("gripper", "moving_jaw_so101_v1"):
        ET.SubElement(contact, "exclude", body1="mallet", body2=body)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def build_tiger_model(config, *, return_xml=False):
    """Reuse exactly the tabletop geometry; move its one mallet into the gripper."""
    import numpy as np
    from .tabletop import build_tabletop
    root = ET.parse(ROOT / "assets/so101/so101_new_calib.xml").getroot()
    root.find("compiler").set("meshdir", str(ROOT / "assets/so101/assets"))
    tabletop = ET.fromstring(build_tabletop(config))
    for name in ("option", "visual"):
        root.append(tabletop.find(name))
    world = root.find("worldbody")
    layout = config.get("robot_layout", {"plate_position": [.28,0,0], "plate_yaw_degrees": 0})
    yaw = np.deg2rad(layout["plate_yaw_degrees"])
    stage = ET.SubElement(world, "body", name="tabletop",
                          pos=" ".join(map(str, layout["plate_position"])),
                          quat=f"{np.cos(yaw/2)} 0 0 {np.sin(yaw/2)}")
    for element in tabletop.find("worldbody"):
        if element.get("name") != "resting_mallet":
            stage.append(element)
    fish = stage.find("body[@name='tiger_fish']")
    fish.set("name", "fish")
    fish.find("geom[@name='fish_lower']").set("name", "fish_base")
    fish.find("geom[@name='fish_shell']").set("name", "fish_target")
    site = fish.find("site[@name='candidate_strike_point']")
    site.set("name", "target")
    # Provisional crown target on the shell, forward of the ears.
    h, d = config["fish"]["height"], config["fish"]["depth"]
    y = -d * .1
    z = h*.565 + h*.355*np.sqrt(1-(y/(d*.47))**2)
    site.set("pos", f"0 {y} {z}")
    gripper = root.find(".//body[@name='gripper']")
    mallet = ET.SubElement(gripper, "body", name="mallet", pos="-0.0079 -0.000218121 -0.0981274")
    m = config["mallet"]
    r = m["handle_diameter"]/2
    length = m["handle_length"]
    centre = length + m["head_length"]*.2
    ET.SubElement(mallet, "geom", name="mallet_handle", type="capsule", fromto=f"0 0 {-r} 0 0 {-length+r}", size=str(r), mass="0.012", rgba="0.60 0.34 0.15 1")
    ET.SubElement(mallet, "geom", name="mallet_head", type="ellipsoid", pos=f"0 0 {-centre}", size=f"{m['head_radius']} {m['head_radius']} {m['head_length']/2}", mass="0.008", rgba="0.93 0.89 0.80 1")
    ET.SubElement(mallet, "site", name="head", pos=f"0 0 {-centre}", size="0.001", group="4")
    cameras = config.get("cameras")
    if cameras:
        def add_camera(parent, name, settings):
            position = np.asarray(settings["position"], dtype=float)
            zaxis = position - np.asarray(settings["look_at"], dtype=float)
            zaxis /= np.linalg.norm(zaxis)
            up = np.array([0., 0., 1.]) if name == "side" else np.array([0., 1., 0.])
            xaxis = np.cross(up, zaxis)
            xaxis /= np.linalg.norm(xaxis)
            yaxis = np.cross(zaxis, xaxis)
            ET.SubElement(parent, "camera", name=name,
                          pos=" ".join(map(str, position)),
                          xyaxes=" ".join(map(str, np.r_[xaxis, yaxis])),
                          fovy=str(settings["fovy"]))
        add_camera(world, "side", cameras["side"])
        parent = root.find(f".//body[@name='{cameras['wrist']['parent_body']}']")
        if parent is None:
            raise ValueError("Wrist camera parent body not found")
        add_camera(parent, "wrist", cameras["wrist"])
    contact = ET.SubElement(root, "contact")
    for body in ("gripper", "moving_jaw_so101_v1"):
        ET.SubElement(contact, "exclude", body1="mallet", body2=body)
    xml = ET.tostring(root, encoding="unicode")
    return xml if return_xml else mujoco.MjModel.from_xml_string(xml)
