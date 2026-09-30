# shinro-demo-lekiwi

The **LeKiwi** reference-robot demo for the [shinro control framework](https://github.com/Shinro-xyz/shinro-python-modules)
— a holonomic 3-wheel base + SO-ARM100 6-DOF arm, simulated in MuJoCo.

This repo holds everything robot-specific that the framework deliberately does
not ship: the vendored MJCF model and mesh assets, the TOML configs and
scenarios, the runnable demos, and the full-loop MuJoCo integration tests. It
plugs into shinro through the framework's public extension seams — a registered
`[physics].preset = "lekiwi"` and the five component ABCs — not framework
internals.

## Layout

```
src/shinro_demo_lekiwi/   Python package: asset paths, helpers, the "lekiwi" preset
  assets/lekiwi-sim/      bundled MJCF + meshes (installed as package data)
configs/                  robot_config.toml + controller/estimator/plant/trajectory TOMLs
scenarios/                complete closed-loop scenario TOMLs
demos/                    runnable demos (`python -m demos.demo_*`)
tests/                    full-loop MuJoCo integration suite
```

## Requirements

Python ≥ 3.12 and a **shinro source checkout** (this repo consumes the framework
from a checkout so it tracks the working tree, not a published release).

> The bundled MuJoCo model and meshes under `src/shinro_demo_lekiwi/assets/`
> are the LeKiwi / SO-ARM100 assets (see the provenance notes in
> `assets/lekiwi-sim/README.md`).

```bash
# Sibling layout — clone both repos next to each other:
#   parent/
#     shinro-python-modules/
#     shinro-demo-lekiwi/

cd shinro-demo-lekiwi
make install          # editable: ../shinro-python-modules + this repo [mujoco,media,dev]
```

`make install` runs:

```bash
pip install -e ../shinro-python-modules
pip install -e ".[mujoco,media,dev]"
```

Override the framework path with `make install SHINRO=/path/to/shinro-python-modules`.

## Run the demos

```bash
python -m demos.demo_simple            # terminal output, no viewer
python -m demos.demo_arm_trajectory    # arm trajectory + live viewer
python -m demos.demo_base_tracking     # base tracking, LQR or MPC + observer
python -m demos.demo_pick_and_place    # pick a block, drive out, place it, drive back
```

### Compiled control

The base tracks a smooth **B-spline** reference (`configs/trajectories/base_bspline.toml`)
from a compiled Zig kernel instead of the Python controller. Two control laws are
compiled:

```bash
shinro build scenarios/base_tracking.toml      --import shinro_demo_lekiwi --out build/compiled_base   # KF + MPC_LTI (base)
shinro build scenarios/base_tracking_mppi.toml --import shinro_demo_lekiwi --out build/compiled_mppi   # KF + MPPI (base)
shinro build scenarios/arm_tracking.toml       --import shinro_demo_lekiwi --out build/compiled_arm    # KF + PID (arm)
python -m demos.demo_compiled_control          # base: [mpc|mppi|all]
python -m demos.demo_compiled_arm              # arm: base still, EE follows a 6-D B-spline
```

The demo loads each `lib<name>.so` (the `shinro_step` C ABI) and drives the
MuJoCo closed loop from it — the host only samples the sensor, packs the input
ports (drawing MPPI's `epsilon` perturbations host-side), and feeds the
recurrent `state_*` ports back. It checks parity against the live Python loop in
lockstep (same inputs each tick): MPC reproduces to QP-solver precision (~1e-6),
MPPI to float exactness (~1e-15), PID to bit exactness (0.0).

Each GIF is a composite (1800x720): a 3-D scene, a **bird's-eye x-y plot** (base
demos) or a **3-D end-effector path plot** (arm demo) of the reference curve vs
the actual trajectory, and a side panel of tracking error vs time (position and,
for the base, heading).

The builds are **ReleaseFast** (`[compile].optimize = "release"`). The kernel is
a plain C-ABI artifact, so it is language-agnostic — `make interop` calls the
same `.so` from C, C++, Zig, and Python (see [`interop/`](./interop/)).

> MPPI's closed-loop trajectory is *not* stable against a per-tick difference:
> its softmax weighting is chaotic, so an independent live run diverges within a
> few dozen ticks even though every tick matches to ~1e-15. Parity is therefore
> measured in lockstep, not across two independent runs.

## Test

```bash
make test    # pytest tests/ — full-loop MuJoCo integration suite
```

## The `lekiwi` preset

A scenario opts into the LeKiwi model by declaring it, instead of the framework
hardcoding it:

```toml
[physics]
preset = "lekiwi"

[sim]
config = "configs/robot_config.toml"
```

`src/shinro_demo_lekiwi/presets.py` registers that name with
`shinro.factories.registry.register_physics_preset`; importing
`shinro_demo_lekiwi` registers it. The preset loads the bundled MJCF, nests the
arm under a free-jointed wheel chassis, and bundles the meshes.

Scripts that import the package get the preset automatically. The `shinro` CLI
does not, so name the module explicitly:

```bash
shinro check scenarios/base_tracking.toml --import shinro_demo_lekiwi
```
