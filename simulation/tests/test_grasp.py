import mujoco
import numpy as np
import pytest
from wooden_fish.grasp import ContactGrasp, rollout


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_contact_lift_tap_release(seed):
    sim = ContactGrasp(seed=seed)
    try:
        assert sim.model.jnt_type[sim.free] == mujoco.mjtJoint.mjJNT_FREE
        assert sim.model.body_parentid[sim.object_body] == 0
        assert sim.model.neq == 0
        assert not np.any(sim.data.xfrc_applied)
        report = list(rollout(sim))[-1]
        assert report["passed"], report
        assert not np.any(sim.data.xfrc_applied)
        assert not np.any(sim.data.qfrc_applied)
    finally:
        sim.close()


def test_open_fingers_do_not_hold_mallet():
    sim = ContactGrasp()
    try:
        before = sim.data.site_xpos[sim.head, 2]
        for _ in range(120):
            sim.tick(closed=False)
        assert before-sim.data.site_xpos[sim.head, 2] > .05
        assert max(sim.pad_forces().values()) < .01
    finally:
        sim.close()
