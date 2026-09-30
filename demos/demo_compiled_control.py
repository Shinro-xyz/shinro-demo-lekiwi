# FILE: demos/demo_compiled_control.py
"""Drive the LeKiwi base-tracking simulation from compiled control artifacts.

Two closed-loop control laws are compiled to Zig kernels, each tracking the same
B-spline reference:

    shinro build scenarios/base_tracking.toml      --import shinro_demo_lekiwi --out build/compiled_base
    shinro build scenarios/base_tracking_mppi.toml --import shinro_demo_lekiwi --out build/compiled_mppi

This demo loads each ``lib<name>.so`` (the ``shinro_step`` C ABI) and runs the
whole MuJoCo closed loop from it. The host only samples the sensor, packs the
input ports (sampling MPPI's ``epsilon`` perturbations host-side), calls the
kernel, and feeds the recurrent ``state_*`` ports back — the Python estimator
and controller are not used to drive the plant.

Parity is checked in **lockstep**: each tick the live Python estimator/controller
compute the same action from the same inputs, and the kernel's action is compared
to it. (Comparing two independent 400-tick runs would be dishonest for MPPI:
its softmax weighting is chaotic, so a per-tick difference at the 1e-12 level
amplifies to O(1) over the run even though every tick matches.)

Usage:  python -m demos.demo_compiled_control [mpc|mppi|all]
Output: build/demos/lekiwi_compiled_<name>.gif
"""
import os

os.environ['MUJOCO_GL'] = 'egl'

import json
import sys
import tomllib
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from shinro.codegen.oracle import load_so, step_so
from shinro.factories import ScenarioFactory
from shinro.utils.config_resolver import resolve_config_path

import shinro_demo_lekiwi  # noqa: F401  (importing the package registers the "lekiwi" preset)
from shinro_demo_lekiwi.viz import TrackingPanels, birdseye_camera, compose_h

HERE = Path(__file__).parent.parent

#: name -> (scenario, artifact dir, kernel stem). Both scenarios track the same
#: B-spline; the controller differs (MPC_LTI vs MPPI).
KERNELS = {
    "mpc": ("scenarios/base_tracking.toml", "build/compiled_base", "libbase"),
    "mppi": ("scenarios/base_tracking_mppi.toml", "build/compiled_mppi", "libmppi"),
}

SEED = 42
CAPTURE_EVERY = 2


def _flat(shape) -> int:
    n = 1
    for d in shape or []:
        n *= d
    return n


def _offsets(ports) -> tuple[dict[str, tuple[int, int]], int]:
    out: dict[str, tuple[int, int]] = {}
    off = 0
    for p in ports:
        size = _flat(p["shape"])
        out[p["name"]] = (off, off + size)
        off += size
    return out, off


def _manifest(artifact: str, kernel: str) -> dict | None:
    root = Path(artifact)
    if not ((root / "graph_data_manifest.json").exists() and (root / "lib" / f"{kernel}.so").exists()):
        return None
    return json.loads((root / "graph_data_manifest.json").read_text())


def run_from_kernel(scenario_path: str, artifact: str, kernel: str, render_path: str | None = None):
    """Run the sim from the .so, checking parity against the live loop in lockstep.

    Returns:
        ``(max_parity, frames)`` — the max per-tick |u_so - u_live| and rendered frames.
    """
    manifest = _manifest(artifact, kernel)
    assert manifest is not None
    lib = load_so(artifact, kernel)
    in_idx, n_in = _offsets(manifest["inputs"])
    out_idx, n_out = _offsets(manifest["outputs"])
    st_idx, n_state = _offsets(manifest["state_outputs"])
    n_u = _flat(next(p for p in manifest["inputs"] if p["name"] == "u_prev")["shape"])
    has_eps = "epsilon" in in_idx
    eps_shape = tuple(next(p["shape"] for p in manifest["inputs"] if p["name"] == "epsilon")) if has_eps else None

    scenario = ScenarioFactory(scenario_path).build()
    sim, plant, est, ctrl, traj = scenario.sim, scenario.plant, scenario.estimator, scenario.controller, scenario.trajectory
    cfg = scenario.config
    dt = float(cfg["scenario"]["dt"])
    n_steps = min(int(float(cfg["scenario"]["duration"]) / dt), len(traj))
    noise = cfg.get("noise", {}).get("measurement")
    std = np.asarray((noise or {}).get("std", 0.0), dtype=np.float64)
    limits = cfg.get("scenario", {}).get("input_limits")
    lo = np.asarray(limits["min"], dtype=np.float64) if limits else np.full(n_u, -1e6)
    hi = np.asarray(limits["max"], dtype=np.float64) if limits else np.full(n_u, 1e6)

    rng = np.random.default_rng(SEED)
    eps_rng = np.random.default_rng(SEED + 1)
    sigma = None
    if has_eps:
        with open(resolve_config_path(cfg["controller"]["config"]), "rb") as f:
            sigma = np.asarray(tomllib.load(f).get("noise_sigma", 1.0), dtype=np.float64)

    renderer = camera = panels = None
    frames = []
    if render_path is not None:
        ref_xy = np.asarray(traj)[:, :2]
        control_points = None
        traj_cfg = cfg.get("trajectory", {}).get("config")
        if traj_cfg:
            with open(resolve_config_path(traj_cfg), "rb") as f:
                control_points = tomllib.load(f).get("control_points")
        renderer = mujoco.Renderer(sim.engine.model, width=720, height=720)
        camera = mujoco.MjvCamera()
        mid = ref_xy.mean(axis=0)
        birdseye_camera(camera, lookat=(float(mid[0]), float(mid[1]), 0.05), distance=3.6)
        panels = TrackingPanels(ref_xy, float(cfg["scenario"]["duration"]), control_points)

    inputs = np.zeros(max(n_in, 1))
    state = np.zeros(max(n_state, 1))
    # Host-seeded KF covariance (P0 = 0.1*I), matching the live KalmanFilter's P.
    p_shape = next(p["shape"] for p in manifest["state_outputs"] if p["name"] == "state_P")
    s0, s1 = st_idx["state_P"]
    state[s0:s1] = (np.eye(p_shape[0]) * 0.1).ravel()
    u_prev = np.zeros(n_u)
    parity = 0.0

    for step in range(n_steps):
        true = np.asarray(plant.get_state(), dtype=np.float64).flatten()
        ref = np.asarray(traj[step], dtype=np.float64).flatten()
        meas = true + rng.normal(0.0, np.broadcast_to(std, true.shape)) if std.any() else true

        # Live reference (lockstep): same measurement, same previous control.
        est_x = np.asarray(est.estimate(meas.reshape(-1, 1), u_prev.reshape(-1, 1)), dtype=np.float64).flatten()

        inputs[:] = 0.0
        eps = None
        for name, (d0, d1) in in_idx.items():
            if name == "y":
                inputs[d0:d1] = meas
            elif name == "x_ref":
                inputs[d0:d1] = ref
            elif name == "u_prev":
                inputs[d0:d1] = u_prev
            elif name == "epsilon":
                k_steps = eps_shape[1] // n_u
                eps = (eps_rng.normal(0.0, 1.0, eps_shape).reshape(eps_shape[0], k_steps, n_u) * sigma).reshape(eps_shape)
                inputs[d0:d1] = eps.ravel()
            elif name.startswith("state_") and name in st_idx:
                a, b = st_idx[name]
                inputs[d0:d1] = state[a:b]

        if has_eps:
            # Align the live controller's recurrent nominal sequence with the
            # kernel's, so the comparison is per-tick *arithmetic*. MPPI's
            # softmax recurrence is chaotic: left independent, a 1e-16 per-tick
            # difference amplifies to O(1) within ~75 ticks, even though every
            # tick computes the same thing from the same inputs.
            if hasattr(ctrl, "u"):
                a, b = st_idx["state_u"]
                ctrl.u = state[a:b].reshape(np.asarray(ctrl.u).shape)
            u_live = np.asarray(ctrl.compute(est_x, ref, epsilon=eps), dtype=np.float64).flatten()
        else:
            u_live = np.asarray(ctrl.compute(est_x, ref), dtype=np.float64).flatten()
        u_live = np.clip(u_live, lo, hi)

        out, state = step_so(lib, inputs, n_out, n_state)
        u = out[out_idx["u"][0]:out_idx["u"][1]]
        parity = max(parity, float(np.max(np.abs(u - u_live))))

        u_prev = u
        plant.step(u)
        sim.step()

        if renderer is not None and step % CAPTURE_EVERY == 0:
            base = np.asarray(sim.base.get_state(), dtype=np.float64)
            panels.update(base[:2], ref[:2], step * dt, float(np.linalg.norm(base[:2] - ref[:2])))
            renderer.update_scene(sim.engine.data, camera)
            frames.append(compose_h([renderer.render(), panels.frame()], height=720))

    if renderer is not None:
        renderer.close()
        Path(render_path).parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(render_path, frames, fps=50 // CAPTURE_EVERY, loop=0, plugin='pillow', optimize=True)
    return parity, frames


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(KERNELS) if which == "all" else [which]

    for name in names:
        scenario_path, artifact, kernel = KERNELS[name]
        manifest = _manifest(artifact, kernel)
        if manifest is None:
            print(f"[{name}] no compiled artifact at {artifact}/ — build it first:")
            print(f"  shinro build {scenario_path} --import shinro_demo_lekiwi --out {artifact}\n")
            continue

        out_path = str(HERE / "build" / "demos" / f"lekiwi_compiled_{name}.gif")
        parity, frames = run_from_kernel(scenario_path, artifact, kernel, out_path)
        so = Path(artifact) / "lib" / f"{kernel}.so"

        print(f"[{name}] {so}  ({manifest.get('nodes_total')} nodes, {so.stat().st_size / 1024:.0f} KiB)")
        print(f"[{name}] compiled kernel drove {len(frames) * CAPTURE_EVERY} sim steps (no Python controller)")
        print(f"[{name}] lockstep parity vs live loop: max |u_so - u_python| = {parity:.2e}")
        print(f"[{name}] ✅ GIF saved: {out_path} ({len(frames)} frames)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
