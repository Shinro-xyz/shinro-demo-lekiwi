# FILE: demos/demo_compiled_arm.py
"""Drive the LeKiwi arm-tracking simulation from a compiled control artifact.

The base stays still — only the end-effector follows a 6-D B-spline path
([x, y, z, roll, pitch, yaw]). The PID + Kalman control law is compiled to a Zig
kernel:

    shinro build scenarios/arm_tracking.toml --import shinro_demo_lekiwi --out build/compiled_arm

As in the base demos, the host only samples the sensor, packs the ports, feeds
the recurrent ``state_*`` ports back, and applies the kernel's Cartesian twist —
the Python estimator/controller are not used to drive the arm. Parity is checked
in lockstep.

Usage:  python -m demos.demo_compiled_arm
Output: build/demos/lekiwi_compiled_arm.gif
"""
import os

os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

import shinro_demo_lekiwi  # noqa: F401  (importing the package registers the "lekiwi" preset)
from shinro.factories import ScenarioFactory
from shinro_demo_lekiwi.host import drive, kernel_info, manifest_for
from shinro_demo_lekiwi.viz import ArmPanels, compose_h

HERE = Path(__file__).parent.parent
SCENARIO = "scenarios/arm_tracking.toml"
ARTIFACT = "build/compiled_arm"
KERNEL = "libarm"
OUTPUT_PATH = str(HERE / "build" / "demos" / "lekiwi_compiled_arm.gif")
CAPTURE_EVERY = 2


def main() -> int:
    if manifest_for(ARTIFACT, KERNEL) is None:
        print(f"no compiled artifact at {ARTIFACT}/ — build it first:")
        print(f"  shinro build {SCENARIO} --import shinro_demo_lekiwi --out {ARTIFACT}")
        return 1

    scenario = ScenarioFactory(SCENARIO).build()
    sim = scenario.sim
    ref = np.asarray(scenario.trajectory)
    duration = float(scenario.config["scenario"]["duration"])

    renderer = mujoco.Renderer(sim.engine.model, width=720, height=720)
    camera = mujoco.MjvCamera()
    camera.distance = 1.1
    camera.azimuth = 130
    camera.elevation = -16
    camera.lookat[:] = [0.08, 0.04, 0.16]
    panels = ArmPanels(ref[:, :3], duration)
    frames = []

    def on_tick(step, dt, state, reference):
        if step % CAPTURE_EVERY != 0:
            return
        ee, rf = state[:3], reference[:3]
        panels.update(
            ee,
            rf,
            step * dt,
            float(np.linalg.norm(ee - rf)),
            float(np.linalg.norm(state[3:6] - reference[3:6])),
        )
        renderer.update_scene(sim.engine.data, camera)
        frames.append(compose_h([renderer.render(), panels.frame()], height=720))

    parity = drive(SCENARIO, ARTIFACT, KERNEL, plant_name="arm", scenario=scenario, on_tick=on_tick)
    renderer.close()

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(OUTPUT_PATH, frames, fps=50 // CAPTURE_EVERY, loop=0, plugin='pillow', optimize=True)

    print(f"kernel: {kernel_info(ARTIFACT, KERNEL)}")
    print(f"compiled kernel drove {len(frames) * CAPTURE_EVERY} sim steps (no Python controller)")
    print(f"lockstep parity vs live loop: max |u_so - u_python| = {parity:.2e}")
    print(f"✅ GIF saved: {OUTPUT_PATH} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
