# FILE: demos/demo_compiled_pickplace.py
"""Compiled pick-and-place: a compiled MPC drives the holonomic base along an
out-and-back arc while a compiled PID arm picks up the block, carries it, sets it
down, and the base returns — all from two loaded ``shinro_step`` kernels.

Build both kernels first:

    shinro build scenarios/pickplace_base.toml --import shinro_demo_lekiwi --out build/compiled_ppbase
    shinro build scenarios/pickplace_arm.toml  --import shinro_demo_lekiwi --out build/compiled_pparm

Usage:  python -m demos.demo_compiled_pickplace
Output: build/demos/lekiwi_compiled_pickplace.gif
"""
import os

os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

import shinro_demo_lekiwi  # noqa: F401  (importing the package registers the "lekiwi" preset)
from shinro.factories import TrajectoryFactory
from shinro.simulation import RobotSim
from shinro.utils.config_resolver import resolve_config_path
from shinro_demo_lekiwi import MESH_DIR, MJCF_PATH, inject_free_joint, load_model_assets
from shinro_demo_lekiwi.host import Channel, drive_channels, kernel_info, manifest_for
from shinro_demo_lekiwi.scene import (
    GROUND_Z,
    block_joint_addresses,
    carry_block,
    inject_graspable_block,
    park_block,
)
from shinro_demo_lekiwi.viz import TrackingPanels, compose_h

HERE = Path(__file__).parent.parent
OUTPUT_PATH = str(HERE / "build" / "demos" / "lekiwi_compiled_pickplace.gif")
DT = 0.02
CAPTURE_EVERY = 2
PICK_POSE = (0.20, -0.03, GROUND_Z)
GRIP_T = 2.6      # jaw closes (arm is down on the block)
RELEASE_T = 11.0  # jaw opens (arm is down at the drop)

BASE_KERNEL = ("build/compiled_ppbase", "libppbase")
ARM_KERNEL = ("build/compiled_pparm", "libpparm")


def main() -> int:
    for artifact, kernel in (BASE_KERNEL, ARM_KERNEL):
        if manifest_for(artifact, kernel) is None:
            print(f"missing {artifact}/lib/{kernel}.so — build both kernels first:")
            print("  shinro build scenarios/pickplace_base.toml --import shinro_demo_lekiwi --out build/compiled_ppbase")
            print("  shinro build scenarios/pickplace_arm.toml  --import shinro_demo_lekiwi --out build/compiled_pparm")
            return 1

    xml = inject_graspable_block(inject_free_joint(Path(MJCF_PATH).read_text()), pos=PICK_POSE)
    sim = RobotSim(
        str(resolve_config_path("configs/robot_config.toml")),
        xml_string=xml,
        assets=load_model_assets(MESH_DIR),
    )
    sim.reset()
    block_qadr, block_dofadr = block_joint_addresses(sim.engine.model)

    base_ref = np.asarray(TrajectoryFactory("configs/trajectories/pickplace_base.toml").create())
    arm_rel = np.asarray(TrajectoryFactory("configs/trajectories/pickplace_arm.toml").create())
    # The arm's end-effector reference is a WORLD pose and the arm is mounted on
    # the base, so its target must move with the base trajectory (the base yaw
    # stays ~0, so a pure translation is enough).
    arm_ref = arm_rel.copy()
    arm_ref[:, 0] += base_ref[:, 0]
    arm_ref[:, 1] += base_ref[:, 1]
    n_steps = min(len(base_ref), len(arm_ref))

    channels = [
        Channel(
            "base", BASE_KERNEL[0], BASE_KERNEL[1], "base",
            "configs/controllers/mpc_lti_base.toml", "configs/estimators/kalman_base.toml",
            base_ref, noise_std=[0.01, 0.01, 0.02],
            input_limits=([-0.5, -0.5, -1.0], [0.5, 0.5, 1.0]),
        ),
        Channel(
            "arm", ARM_KERNEL[0], ARM_KERNEL[1], "arm",
            "configs/controllers/pid_arm.toml", "configs/estimators/kalman_arm.toml",
            arm_ref, noise_std=[0.005, 0.005, 0.005, 0.01, 0.01, 0.01],
            input_limits=([-0.5] * 6, [0.5] * 6),
        ),
    ]

    renderer = mujoco.Renderer(sim.engine.model, width=720, height=720)
    camera = mujoco.MjvCamera()
    camera.distance = 2.0
    camera.azimuth = 128
    camera.elevation = -30
    camera.lookat[:] = [0.4, 0.0, 0.1]
    panels = TrackingPanels(base_ref[:, :2], n_steps * DT)
    frames = []
    phase = {"p": "approach", "drop": None}

    sim.engine.set_joint_ctrl("Jaw", 0.0)  # start open

    def on_tick(step, dt, sim):
        t = step * dt
        # One-shot transitions (a threshold check would re-grasp every tick).
        if phase["p"] == "approach" and t >= GRIP_T:
            phase["p"] = "carry"
        elif phase["p"] == "carry" and t >= RELEASE_T:
            phase["p"] = "done"
            ee = np.asarray(sim.arm._get_ee_pos(), dtype=np.float64)
            phase["drop"] = (float(ee[0]), float(ee[1]), GROUND_Z)

        sim.engine.set_joint_ctrl("Jaw", 0.5 if phase["p"] == "carry" else 0.0)

        if phase["p"] == "carry":
            carry_block(sim, block_qadr, block_dofadr)
        else:
            park_block(sim, block_qadr, block_dofadr, phase["drop"] or PICK_POSE)

        if step % CAPTURE_EVERY == 0:
            base = np.asarray(sim.base.get_state(), dtype=np.float64)
            ref = base_ref[step]
            panels.update(
                base[:2], ref[:2], t,
                float(np.linalg.norm(base[:2] - ref[:2])),
                float(abs(base[2] - ref[2])),
            )
            camera.lookat[:] = [float(base[0]), float(base[1]), 0.1]
            renderer.update_scene(sim.engine.data, camera)
            frames.append(compose_h([renderer.render(), panels.frame()], height=720))

    parities = drive_channels(sim, channels, dt=DT, on_tick=on_tick)
    renderer.close()

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(OUTPUT_PATH, frames, fps=50 // CAPTURE_EVERY, loop=0, plugin='pillow', optimize=True)

    block_final = np.asarray(sim.engine.data.qpos[block_qadr:block_qadr + 3]).copy()
    print(f"base kernel: {kernel_info(*BASE_KERNEL)}")
    print(f"arm  kernel: {kernel_info(*ARM_KERNEL)}")
    print(f"ran {n_steps} steps — MPC base + PID arm, both from .so (no Python controller drove the plant)")
    for name, p in parities.items():
        print(f"  [{name}] lockstep parity vs live loop: max |u_so - u_python| = {p:.2e}")
    print(f"  block: picked at x={PICK_POSE[0]:+.3f} -> placed at x={block_final[0]:+.3f} m")
    print(f"✅ GIF saved: {OUTPUT_PATH} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
