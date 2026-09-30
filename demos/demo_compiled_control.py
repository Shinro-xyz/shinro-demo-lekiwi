# FILE: demos/demo_compiled_control.py
"""Drive the LeKiwi base-tracking simulation from a compiled control artifact.

The scenario's KF+LQR control law is compiled once to a Zig kernel:

    shinro build scenarios/base_tracking.toml --import shinro_demo_lekiwi --out build/compiled_base

This demo then loads ``build/compiled_base/lib/libbase.so`` (the ``shinro_step``
C ABI) and runs the whole MuJoCo closed loop from it. The host only samples the
sensor, packs the input ports, calls the kernel, and feeds the recurrent
``state_*`` ports back — the Python estimator and controller are not used in that
loop at all. It finishes by re-running the scenario with the live Python
components and reporting the control parity.

Usage:  python -m demos.demo_compiled_control
Output: build/demos/lekiwi_compiled_control.gif
"""
import os

os.environ['MUJOCO_GL'] = 'egl'

import json
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from shinro.codegen.oracle import load_so, step_so
from shinro.factories import ScenarioFactory
from shinro.simulation.runner import run_scenario

import shinro_demo_lekiwi  # noqa: F401  (importing the package registers the "lekiwi" preset)

HERE = Path(__file__).parent.parent
SCENARIO = "scenarios/base_tracking.toml"
ARTIFACT = HERE / "build" / "compiled_base"
OUTPUT_PATH = str(HERE / "build" / "demos" / "lekiwi_compiled_control.gif")


def _flat(shape) -> int:
    n = 1
    for d in shape or []:
        n *= d
    return n


def _offsets(ports) -> tuple[dict[str, tuple[int, int]], int]:
    """Map each port name to its (start, stop) in the flat C-ABI buffer."""
    out: dict[str, tuple[int, int]] = {}
    off = 0
    for p in ports:
        size = _flat(p["shape"])
        out[p["name"]] = (off, off + size)
        off += size
    return out, off


def _load_manifest() -> dict:
    manifest = ARTIFACT / "graph_data_manifest.json"
    if not (manifest.exists() and (ARTIFACT / "lib" / "libbase.so").exists()):
        raise SystemExit(
            f"no compiled artifact at {ARTIFACT}\n"
            "build it first:\n"
            f"  shinro build {SCENARIO} --import shinro_demo_lekiwi --out build/compiled_base"
        )
    return json.loads(manifest.read_text())


def run_from_kernel(manifest: dict, render_path: str | None = None):
    """Run the scenario's sim with control from the .so; return (controls, frames)."""
    lib = load_so(ARTIFACT, "libbase")
    in_idx, n_in = _offsets(manifest["inputs"])
    st_idx, n_state = _offsets(manifest["state_outputs"])
    n_out = sum(_flat(p["shape"]) for p in manifest["outputs"])
    n_u = _flat(next(p for p in manifest["inputs"] if p["name"] == "u_prev")["shape"])

    scenario = ScenarioFactory(SCENARIO).build()
    sim, plant, traj = scenario.sim, scenario.plant, scenario.trajectory
    cfg = scenario.config
    dt = float(cfg["scenario"]["dt"])
    n_steps = min(int(float(cfg["scenario"]["duration"]) / dt), len(traj))

    noise = cfg.get("noise", {}).get("measurement")
    seed = (noise or {}).get("seed", 42)
    rng = np.random.default_rng(seed)
    std = np.asarray((noise or {}).get("std", 0.0), dtype=np.float64)

    renderer = camera = None
    frames = []
    if render_path is not None:
        renderer = mujoco.Renderer(sim.engine.model, width=400, height=300)
        camera = mujoco.MjvCamera()
        camera.distance = 1.6
        camera.azimuth = 135
        camera.elevation = -20
        camera.lookat[:] = [0.0, 0.0, 0.1]

    inputs = np.zeros(max(n_in, 1))
    state = np.zeros(max(n_state, 1))
    # The KF covariance is a host-seeded recurrent port: seed P0 = 0.1*I (the
    # live KalmanFilter's initial P), or the kernel starts from a wrong gain.
    p_shape = next(p["shape"] for p in manifest["state_outputs"] if p["name"] == "state_P")
    s0, s1 = st_idx["state_P"]
    state[s0:s1] = (np.eye(p_shape[0]) * 0.1).ravel()
    u_prev = np.zeros(n_u)
    controls = []

    for step in range(n_steps):
        true = np.asarray(plant.get_state(), dtype=np.float64).flatten()
        ref = np.asarray(traj[step], dtype=np.float64).flatten()
        meas = true + rng.normal(0.0, np.broadcast_to(std, true.shape)) if noise else true

        inputs[:] = 0.0
        for name, value in (("y", meas), ("x_ref", ref), ("u_prev", u_prev)):
            a, b = in_idx[name]
            inputs[a:b] = value
        # Recurrent state: feed the previous tick's state_* outputs back in.
        for name in ("state_x_hat", "state_P"):
            if name in in_idx:
                (s0, s1), (d0, d1) = st_idx[name], in_idx[name]
                inputs[d0:d1] = state[s0:s1]

        out, state = step_so(lib, inputs, n_out, n_state)
        u = out[:n_u]
        controls.append(u.copy())
        u_prev = u

        plant.step(u)
        sim.step()

        if renderer is not None and step % 2 == 0:
            base = np.asarray(sim.base.get_state())
            camera.lookat[:] = [float(base[0]), 0.0, 0.1]
            renderer.update_scene(sim.engine.data, camera)
            frames.append(renderer.render())

    if renderer is not None:
        renderer.close()
        Path(render_path).parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(render_path, frames, fps=25, loop=0, plugin='pillow', optimize=True)

    return np.array(controls), frames


def main() -> int:
    manifest = _load_manifest()
    so = ARTIFACT / "lib" / "libbase.so"
    print(
        f"kernel: {so}  ({manifest.get('nodes_total')} nodes, "
        f"{manifest.get('buf_bytes')} B workspace, {so.stat().st_size / 1024:.0f} KiB)"
    )

    controls_so, frames = run_from_kernel(manifest, OUTPUT_PATH)

    # Live Python reference: same scenario, same seed, same measurement noise.
    live = run_scenario(ScenarioFactory(SCENARIO).build(), seed=42)
    controls_live = np.array([r.control for r in live.records])
    n = min(len(controls_so), len(controls_live))
    parity = float(np.max(np.abs(controls_so[:n] - controls_live[:n])))

    print(f"compiled kernel drove {len(controls_so)} sim steps (no Python controller)")
    print(f"parity vs live Python loop: max |u_so - u_python| = {parity:.2e}")
    print(f"✅ GIF saved: {OUTPUT_PATH} ({len(frames)} frames, 400x300, 25 fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
