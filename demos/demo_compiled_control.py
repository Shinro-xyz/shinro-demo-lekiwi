# FILE: demos/demo_compiled_control.py
"""Drive the LeKiwi base-tracking simulation from compiled control artifacts.

Two closed-loop control laws are compiled to Zig kernels, each tracking the same
B-spline reference:

    shinro build scenarios/base_tracking.toml      --import shinro_demo_lekiwi --out build/compiled_base
    shinro build scenarios/base_tracking_mppi.toml --import shinro_demo_lekiwi --out build/compiled_mppi

The host driver (shinro_demo_lekiwi.host) samples the sensor, packs the ports
(drawing MPPI's epsilon perturbations host-side), feeds the recurrent state_*
ports back, and applies the kernel's control — the Python estimator/controller
are not used to drive the plant. Parity is checked in lockstep.

Each GIF is a composite: a top-down scene, a bird's-eye x-y plot of the B-spline
control polygon + reference curve + the base's actual path, and a tracking-error
panel (position and heading).

Usage:  python -m demos.demo_compiled_control [mpc|mppi|all]
Output: build/demos/lekiwi_compiled_<name>.gif
"""
import os

os.environ['MUJOCO_GL'] = 'egl'

import sys
import tomllib
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

import shinro_demo_lekiwi  # noqa: F401  (importing the package registers the "lekiwi" preset)
from shinro.factories import ScenarioFactory
from shinro.utils.config_resolver import resolve_config_path
from shinro_demo_lekiwi.host import drive, kernel_info, manifest_for
from shinro_demo_lekiwi.viz import TrackingPanels, birdseye_camera, compose_h

HERE = Path(__file__).parent.parent

#: name -> (scenario, artifact dir, kernel stem). Both scenarios track the same
#: B-spline; the controller differs (MPC_LTI vs MPPI).
KERNELS = {
    "mpc": ("scenarios/base_tracking.toml", "build/compiled_base", "libbase"),
    "mppi": ("scenarios/base_tracking_mppi.toml", "build/compiled_mppi", "libmppi"),
}
CAPTURE_EVERY = 2


def run_from_kernel(scenario_path: str, artifact: str, kernel: str, render_path: str | None = None):
    """Run the sim from the .so; return (lockstep parity, rendered frames)."""
    scenario = ScenarioFactory(scenario_path).build()
    sim = scenario.sim
    ref = np.asarray(scenario.trajectory)
    ref_xy = ref[:, :2]
    duration = float(scenario.config["scenario"]["duration"])

    control_points = None
    traj_cfg = scenario.config.get("trajectory", {}).get("config")
    if traj_cfg:
        with open(resolve_config_path(traj_cfg), "rb") as f:
            control_points = tomllib.load(f).get("control_points")

    renderer = camera = panels = None
    frames = []
    if render_path is not None:
        renderer = mujoco.Renderer(sim.engine.model, width=720, height=720)
        camera = mujoco.MjvCamera()
        mid = ref_xy.mean(axis=0)
        birdseye_camera(camera, lookat=(float(mid[0]), float(mid[1]), 0.05), distance=3.6)
        panels = TrackingPanels(ref_xy, duration, control_points)

    def on_tick(step, dt, state, reference):
        if render_path is None or step % CAPTURE_EVERY != 0:
            return
        panels.update(
            state[:2],
            reference[:2],
            step * dt,
            float(np.linalg.norm(state[:2] - reference[:2])),
            float(abs(state[2] - reference[2])),
        )
        renderer.update_scene(sim.engine.data, camera)
        frames.append(compose_h([renderer.render(), panels.frame()], height=720))

    parity = drive(scenario_path, artifact, kernel, plant_name="base", scenario=scenario, on_tick=on_tick)

    if render_path is not None:
        renderer.close()
        Path(render_path).parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(render_path, frames, fps=50 // CAPTURE_EVERY, loop=0, plugin='pillow', optimize=True)

    return parity, frames


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(KERNELS) if which == "all" else [which]

    for name in names:
        scenario_path, artifact, kernel = KERNELS[name]
        if manifest_for(artifact, kernel) is None:
            print(f"[{name}] no compiled artifact at {artifact}/ — build it first:")
            print(f"  shinro build {scenario_path} --import shinro_demo_lekiwi --out {artifact}\n")
            continue

        out_path = str(HERE / "build" / "demos" / f"lekiwi_compiled_{name}.gif")
        parity, frames = run_from_kernel(scenario_path, artifact, kernel, out_path)

        print(f"[{name}] {kernel_info(artifact, kernel)}")
        print(f"[{name}] compiled kernel drove {len(frames) * CAPTURE_EVERY} sim steps (no Python controller)")
        print(f"[{name}] lockstep parity vs live loop: max |u_so - u_python| = {parity:.2e}")
        print(f"[{name}] ✅ GIF saved: {out_path} ({len(frames)} frames)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
