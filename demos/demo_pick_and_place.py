# FILE: demos/demo_pick_and_place.py
"""Pick-and-place demo: the arm reaches forward to a block, picks it up, the base
drives it out, the arm sets it down, and the base returns to the start.

The schedule (configs/trajectories/pick_and_place.toml) is the control:
reach forward -> grip -> lift -> drive out -> set down -> release -> retract ->
drive back. The base starts and ends at its origin; the block starts in front of
the robot and is left at the drop site.

The block is a kinematic payload: it is parked on the ground, pinned to the
gripper while the jaw is closed, and parked at the drop site on release — the
host owns the discrete grasp event (no physics weld).

Usage:  python -m demos.demo_pick_and_place
Output: build/demos/lekiwi_pick_and_place.gif
"""
import os

os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from shinro.factories import TrajectoryFactory
from shinro.simulation import RobotSim
from shinro.utils.config_resolver import resolve_config_path
from shinro_demo_lekiwi import MESH_DIR, MJCF_PATH, inject_free_joint, load_model_assets
from shinro_demo_lekiwi.scene import (
    GROUND_Z,
    block_joint_addresses,
    carry_block,
    inject_graspable_block,
    park_block,
)

HERE = Path(__file__).parent.parent
OUTPUT_PATH = str(HERE / "build" / "demos" / "lekiwi_pick_and_place.gif")

#: Where the block rests on the ground, in front of the robot.
PICK_POSE = (0.20, -0.03, GROUND_Z)

# ── Create the sim: free-jointed base + a kinematic block in front ───────
xml = inject_graspable_block(inject_free_joint(Path(MJCF_PATH).read_text()), pos=PICK_POSE)
sim = RobotSim(
    str(resolve_config_path("configs/robot_config.toml")),
    xml_string=xml,
    assets=load_model_assets(MESH_DIR),
)
sim.reset()

model, data = sim.engine.model, sim.engine.data
block_qadr, block_dofadr = block_joint_addresses(model)

# ── Schedule ─────────────────────────────────────────────────────────────
sched = TrajectoryFactory(str(resolve_config_path("configs/trajectories/pick_and_place.toml"))).create()
total_steps = len(sched["arm"])

# ── Renderer ─────────────────────────────────────────────────────────────
renderer = mujoco.Renderer(model, width=400, height=300)
camera = mujoco.MjvCamera()
camera.distance = 1.6
camera.azimuth = 135
camera.elevation = -20
camera.lookat[:] = [0.0, 0.0, 0.1]

frames = []
capture_every = 2
grasped = False
drop_pose = None

for step in range(total_steps):
    arm_twist = sched["arm"][step]
    base_vel = sched["base"][step]
    jaw = float(np.asarray(sched["jaw"][step]).ravel()[0])

    sim.arm.step(arm_twist)
    sim.base.step(base_vel)
    sim.engine.set_joint_ctrl("Jaw", jaw)

    # Host owns the discrete grasp event (jaw 0.5 == closed, 0.0 == open).
    if not grasped and jaw > 0.4:
        grasped = True
    elif grasped and jaw < 0.1:
        grasped = False
        ee = np.asarray(sim.arm._get_ee_pos(), dtype=np.float64)
        drop_pose = (float(ee[0]), float(ee[1]), GROUND_Z)

    sim.step()

    # Move the payload: carried while grasped, else parked where it belongs.
    if grasped:
        carry_block(sim, block_qadr, block_dofadr)
    elif drop_pose is not None:
        park_block(sim, block_qadr, block_dofadr, drop_pose)
    else:
        park_block(sim, block_qadr, block_dofadr, PICK_POSE)

    base_pose = sim.base.get_state()
    camera.lookat[:] = [float(base_pose[0]), 0.0, 0.1]
    if step % capture_every == 0:
        renderer.update_scene(data, camera)
        frames.append(renderer.render())

renderer.close()

block_final = np.asarray(data.qpos[block_qadr:block_qadr + 3]).copy()
base_final = np.asarray(sim.base.get_state()).copy()

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
iio.imwrite(
    OUTPUT_PATH, frames,
    fps=50 // capture_every, loop=0,
    plugin='pillow', optimize=True,
)
print("Pick-and-place complete:")
print(f"  base  : start x=+0.000  ->  end x={base_final[0]:+.3f} m")
print(f"  block : picked at x={PICK_POSE[0]:+.3f}  ->  placed at x={block_final[0]:+.3f} m")
print(f"✅ GIF saved: {OUTPUT_PATH} ({len(frames)} frames, 400x300, {50 // capture_every} fps)")
