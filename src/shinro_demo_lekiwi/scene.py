"""Scene helpers specific to the LeKiwi demo (not part of the physics preset).

The preset supplies the bare robot world; the pick-and-place demo adds a block
on top. Keeping this out of the preset means scenarios that do not need a block
(base/arm tracking) don't get one.

The block is a **kinematic payload**: it has a free joint (so its ``qpos`` can be
written) but its geoms collide with nothing, and the host moves it — parked on
the ground until grasped, pinned to the gripper while carried, parked at the
drop site on release. This avoids the arm sweeping a physics block out of the
way before the grasp, and needs no weld equality (whose reference pose is fixed
at compile time and would fight a moving grasp).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

#: End-effector body the gripper is attached to (matches configs/plants/armrobot.toml).
EE_BODY = "Moving_Jaw_08d-v1"

#: Ground rest height (m) for the default block size.
GROUND_Z = 0.025


def inject_graspable_block(
    xml_string: str,
    pos: tuple[float, float, float] = (0.20, -0.03, GROUND_Z),
    half: tuple[float, float, float] = (0.02, 0.02, 0.02),
    mass: float = 0.05,
    rgba: str = "0.95 0.55 0.15 1",
) -> str:
    """Add a kinematic-payload block (free joint, no collisions) to an MJCF string.

    Args:
        xml_string: MJCF document (already free-jointed) as text.
        pos: Block centre in world coordinates (metres).
        half: Block half-extents (metres).
        mass: Block mass (kg) — cosmetic; it is never integrated.
        rgba: Block colour.

    Returns:
        The MJCF document with the block body added.
    """
    root = ET.fromstring(xml_string)

    worldbody = root.find(".//worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no <worldbody>")

    block = ET.SubElement(worldbody, "body")
    block.set("name", "block")
    block.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
    freejoint = ET.SubElement(block, "freejoint")
    freejoint.set("name", "block_free")
    geom = ET.SubElement(block, "geom")
    geom.set("name", "block_geom")
    geom.set("type", "box")
    geom.set("size", f"{half[0]} {half[1]} {half[2]}")
    geom.set("mass", str(mass))
    geom.set("rgba", rgba)
    # Kinematic payload: never collides (the host owns its pose).
    geom.set("contype", "0")
    geom.set("conaffinity", "0")

    return ET.tostring(root, encoding="unicode")


def block_joint_addresses(model, joint: str = "block_free") -> tuple[int, int]:
    """Return the ``(qpos_adr, dof_adr)`` of the block's free joint."""
    import mujoco

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if jid < 0:
        raise ValueError(f"MJCF has no joint named {joint!r}")
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def _set_block(sim, qpos_adr: int, dof_adr: int, xyz) -> None:
    data = sim.engine.data
    data.qpos[qpos_adr:qpos_adr + 3] = xyz
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[dof_adr:dof_adr + 6] = 0.0


def park_block(sim, qpos_adr: int, dof_adr: int, xyz) -> None:
    """Place the block at a fixed world pose (upright), e.g. on the ground."""
    _set_block(sim, qpos_adr, dof_adr, np.asarray(xyz, dtype=np.float64))


def carry_block(sim, qpos_adr: int, dof_adr: int, plant: str = "arm") -> None:
    """Pin the carried block to a plant's end-effector (host-owned grasp).

    Args:
        sim: A composed :class:`~shinro.simulation.robotsim.RobotSim`.
        qpos_adr: Block free-joint qpos address (see :func:`block_joint_addresses`).
        dof_adr: Block free-joint dof address.
        plant: Name of the arm plant whose end-effector carries the block.
    """
    ee = np.asarray(sim.get_plant(plant)._get_ee_pos(), dtype=np.float64)
    _set_block(sim, qpos_adr, dof_adr, ee)
