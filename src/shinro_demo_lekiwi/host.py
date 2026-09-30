"""Host-side driver for the compiled-control demos.

Runs a scenario's MuJoCo sim from a compiled ``shinro_step`` kernel: the host
samples the sensor, packs the input ports (drawing MPPI's ``epsilon`` host-side),
feeds the recurrent ``state_*`` ports back, and applies the kernel's control.

Parity is checked in **lockstep** — each tick the live Python estimator and
controller compute the same action from the same inputs, and the kernel's action
is compared to it. (Comparing two independent runs would be dishonest for a
chaotic recurrent controller; see the MPPI note in the README.)
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from shinro.codegen.oracle import load_so, step_so
from shinro.factories import ScenarioFactory
from shinro.factories.controller_factory import ControllerFactory
from shinro.factories.estimator_factory import EstimatorFactory
from shinro.utils.config_resolver import resolve_config_path

import shinro_demo_lekiwi  # noqa: F401  (registers the "lekiwi" preset)

SEED = 42


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


def manifest_for(artifact: str, kernel: str) -> dict | None:
    """Return the graph manifest for a built artifact, or ``None`` if not built."""
    root = Path(artifact)
    if not ((root / "graph_data_manifest.json").exists() and (root / "lib" / f"{kernel}.so").exists()):
        return None
    return json.loads((root / "graph_data_manifest.json").read_text())


def kernel_info(artifact: str, kernel: str) -> str:
    so = Path(artifact) / "lib" / f"{kernel}.so"
    n = (manifest_for(artifact, kernel) or {}).get("nodes_total")
    return f"{so}  ({n} nodes, {so.stat().st_size / 1024:.0f} KiB)"


def drive(
    scenario_path: str,
    artifact: str,
    kernel: str,
    *,
    plant_name: str = "base",
    seed: int = SEED,
    scenario=None,
    on_tick: Callable[[int, float, np.ndarray, np.ndarray], None] | None = None,
) -> float:
    """Drive the scenario's sim from the kernel; return the max lockstep parity.

    Args:
        scenario_path: Scenario TOML.
        artifact: Directory holding ``lib/<kernel>.so`` + ``graph_data_manifest.json``.
        kernel: Kernel stem (e.g. ``libbase``).
        plant_name: Which sim plant the kernel controls.
        seed: Measurement-noise seed (matched between the kernel and live loops).
        scenario: A pre-built scenario (so a caller can read its trajectory first);
            built from ``scenario_path`` when omitted.
        on_tick: Called ``(step, dt, plant_state, reference)`` after each tick.
    """
    manifest = manifest_for(artifact, kernel)
    if manifest is None:
        raise FileNotFoundError(f"no compiled artifact at {artifact}/ — build it first")
    lib = load_so(artifact, kernel)
    in_idx, n_in = _offsets(manifest["inputs"])
    out_idx, n_out = _offsets(manifest["outputs"])
    st_idx, n_state = _offsets(manifest["state_outputs"])
    n_u = _flat(next(p for p in manifest["inputs"] if p["name"] == "u_prev")["shape"])
    has_eps = "epsilon" in in_idx
    eps_shape = tuple(next(p["shape"] for p in manifest["inputs"] if p["name"] == "epsilon")) if has_eps else None

    scenario = ScenarioFactory(scenario_path).build() if scenario is None else scenario
    sim = scenario.sim
    plant = sim.get_plant(plant_name)
    est, ctrl, traj = scenario.estimator, scenario.controller, scenario.trajectory
    cfg = scenario.config
    dt = float(cfg["scenario"]["dt"])
    n_steps = min(int(float(cfg["scenario"]["duration"]) / dt), len(traj))
    noise = cfg.get("noise", {}).get("measurement")
    std = np.asarray((noise or {}).get("std", 0.0), dtype=np.float64)
    limits = cfg.get("scenario", {}).get("input_limits")
    lo = np.asarray(limits["min"], dtype=np.float64) if limits else np.full(n_u, -1e6)
    hi = np.asarray(limits["max"], dtype=np.float64) if limits else np.full(n_u, 1e6)

    rng = np.random.default_rng(seed)
    eps_rng = np.random.default_rng(seed + 1)
    sigma = None
    if has_eps:
        with open(resolve_config_path(cfg["controller"]["config"]), "rb") as f:
            sigma = np.asarray(tomllib.load(f).get("noise_sigma", 1.0), dtype=np.float64)

    inputs = np.zeros(max(n_in, 1))
    state = np.zeros(max(n_state, 1))
    if "state_P" in st_idx:  # host-seeded KF covariance (P0 = 0.1*I)
        p_shape = next(p["shape"] for p in manifest["state_outputs"] if p["name"] == "state_P")
        a, b = st_idx["state_P"]
        state[a:b] = (np.eye(p_shape[0]) * 0.1).ravel()
    u_prev = np.zeros(n_u)
    parity = 0.0
    ctrl_u_shape = np.asarray(getattr(ctrl, "u", np.zeros(1))).shape if has_eps else None

    for step in range(n_steps):
        true = np.asarray(plant.get_state(), dtype=np.float64).flatten()
        ref = np.asarray(traj[step], dtype=np.float64).flatten()
        meas = true + rng.normal(0.0, np.broadcast_to(std, true.shape)) if std.any() else true
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
                k = eps_shape[1] // n_u
                eps = (eps_rng.normal(0.0, 1.0, eps_shape).reshape(eps_shape[0], k, n_u) * sigma).reshape(eps_shape)
                inputs[d0:d1] = eps.ravel()
            elif name.startswith("state_") and name in st_idx:
                a, b = st_idx[name]
                inputs[d0:d1] = state[a:b]

        if has_eps:
            # Align the live controller's recurrent nominal with the kernel's so the
            # comparison is per-tick arithmetic (MPPI's softmax recurrence is
            # chaotic and amplifies a 1e-16 difference to O(1) within ~75 ticks).
            if ctrl_u_shape is not None:
                a, b = st_idx["state_u"]
                ctrl.u = state[a:b].reshape(ctrl_u_shape)
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

        if on_tick is not None:
            on_tick(step, dt, np.asarray(plant.get_state(), dtype=np.float64), ref)

    return parity


@dataclass
class Channel:
    """One compiled controller driving one plant of a shared sim."""

    name: str
    artifact: str
    kernel: str
    plant_name: str
    controller_config: str
    estimator_config: str
    reference: np.ndarray
    noise_std: np.ndarray | None = None
    input_limits: tuple = None
    _u: np.ndarray = field(default=None, repr=False)


def drive_channels(
    sim,
    channels: list[Channel],
    *,
    dt: float,
    seed: int = SEED,
    on_tick: Callable[[int, float, object], None] | None = None,
) -> dict[str, float]:
    """Drive several plants of one sim from several compiled kernels.

    Each channel owns its kernel, its live estimator/controller (for the lockstep
    parity), and its reference; every tick all channels compute and apply their
    control, then the sim advances once. Returns ``{channel: max parity}``.

    Args:
        sim: A built :class:`RobotSim` (all plants already present).
        channels: The channels to run.
        dt: Control period.
        seed: Measurement-noise seed (matched between kernel and live loops).
        on_tick: Called ``(step, dt, sim)`` after each sim step (for jaw/grasp
            events and rendering).
    """
    prepared = []
    for ch in channels:
        manifest = manifest_for(ch.artifact, ch.kernel)
        if manifest is None:
            raise FileNotFoundError(f"no compiled artifact for channel '{ch.name}' at {ch.artifact}/")
        in_idx, n_in = _offsets(manifest["inputs"])
        out_idx, n_out = _offsets(manifest["outputs"])
        st_idx, n_state = _offsets(manifest["state_outputs"])
        n_u = _flat(next(p for p in manifest["inputs"] if p["name"] == "u_prev")["shape"])
        plant = sim.get_plant(ch.plant_name)
        est = EstimatorFactory(ch.estimator_config).create(plant=plant, derive_model=True)
        ctrl = ControllerFactory(ch.controller_config).create(plant=plant, derive_model=True)
        if hasattr(ctrl, "attach_plant"):
            ctrl.attach_plant(plant)
        state = np.zeros(max(n_state, 1))
        if "state_P" in st_idx:
            p_shape = next(p["shape"] for p in manifest["state_outputs"] if p["name"] == "state_P")
            a, b = st_idx["state_P"]
            state[a:b] = (np.eye(p_shape[0]) * 0.1).ravel()
        lim = ch.input_limits
        lo = np.asarray(lim[0], dtype=np.float64) if lim else np.full(n_u, -1e6)
        hi = np.asarray(lim[1], dtype=np.float64) if lim else np.full(n_u, 1e6)
        std = np.asarray(ch.noise_std if ch.noise_std is not None else np.zeros(n_u), dtype=np.float64)
        prepared.append({
            "ch": ch, "lib": load_so(ch.artifact, ch.kernel), "plant": plant, "est": est, "ctrl": ctrl,
            "in_idx": in_idx, "out_idx": out_idx, "st_idx": st_idx,
            "n_in": n_in, "n_out": n_out, "n_state": n_state, "n_u": n_u,
            "state": state, "u_prev": np.zeros(n_u), "lo": lo, "hi": hi, "std": std,
        })
    rngs = [np.random.default_rng(seed + i) for i in range(len(channels))]
    n_steps = min(len(ch.reference) for ch in channels)
    parities = {ch.name: 0.0 for ch in channels}

    for step in range(n_steps):
        for pr, ch, rng in zip(prepared, channels, rngs):
            true = np.asarray(pr["plant"].get_state(), dtype=np.float64).flatten()
            ref = np.asarray(ch.reference[step], dtype=np.float64).flatten()
            meas = true + rng.normal(0.0, np.broadcast_to(pr["std"], true.shape)) if pr["std"].any() else true
            est_x = np.asarray(
                pr["est"].estimate(meas.reshape(-1, 1), pr["u_prev"].reshape(-1, 1)), dtype=np.float64
            ).flatten()

            inputs = np.zeros(max(pr["n_in"], 1))
            for name, (d0, d1) in pr["in_idx"].items():
                if name == "y":
                    inputs[d0:d1] = meas
                elif name == "x_ref":
                    inputs[d0:d1] = ref
                elif name == "u_prev":
                    inputs[d0:d1] = pr["u_prev"]
                elif name.startswith("state_") and name in pr["st_idx"]:
                    a, b = pr["st_idx"][name]
                    inputs[d0:d1] = pr["state"][a:b]

            out, pr["state"] = step_so(pr["lib"], inputs, pr["n_out"], pr["n_state"])
            u = out[pr["out_idx"]["u"][0]:pr["out_idx"]["u"][1]]
            u_live = np.asarray(pr["ctrl"].compute(est_x, ref), dtype=np.float64).flatten()
            u_live = np.clip(u_live, pr["lo"], pr["hi"])
            parities[ch.name] = max(parities[ch.name], float(np.max(np.abs(u - u_live))))
            pr["u_prev"] = u
            pr["_u"] = u

        for pr in prepared:
            pr["plant"].step(pr["_u"])
        sim.step()
        if on_tick is not None:
            on_tick(step, dt, sim)

    return parities
