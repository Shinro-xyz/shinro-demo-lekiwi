# shinro-demo-lekiwi

A **LeKiwi** reference-robot demo for the [shinro control framework](https://github.com/Shinro-xyz/shinro-python-modules):
a holonomic 3-wheel base + SO-ARM100 6-DOF arm, simulated in MuJoCo.

The point of this repo is to show shinro's **compiled control** end to end on a
real robot model: a control law (estimator + controller) written as ordinary
Python is compiled to a tiny language-agnostic `.so`, and these demos drive the
MuJoCo simulation **from that compiled kernel**. Everything robot-specific the
framework deliberately does not ship lives here — the vendored MJCF + meshes, the
configs and scenarios, the demos, and the integration tests. It plugs in through
[public extension seams](#the-lekiwi-preset), not framework internals.

## Results at a glance

Every compiled demo drives the sim from the `.so` and compares the kernel's
control to the live Python loop **in lockstep** (identical inputs each tick). The
kernel is a pure function of its ports, so this is the honest correctness check:

| demo | plant | control law (compiled) | kernel | parity vs live Python |
| ---- | ----- | ---------------------- | ------ | --------------------- |
| `demo_compiled_control mpc`   | base | Kalman + **MPC** | 52 nodes · 275 KiB | **1.4e-6** (QP-solver precision) |
| `demo_compiled_control mppi`  | base | Kalman + **MPPI** | 494 nodes · 504 KiB | **2.7e-15** (float-exact) |
| `demo_compiled_arm`           | arm  | Kalman + **PID** | 76 nodes · 218 KiB | **0.0** (bit-exact) |
| `demo_compiled_pickplace`     | base + arm | **MPC** + **PID** | 52 + 76 nodes | **1.9e-6** / **0.0** |

All built **ReleaseFast** (`[compile].optimize = "release"`) — a few hundred KiB
each, versus ~11 MB for the debug build, with the same numerics. The kernels are
plain C-ABI shared objects: `make interop` calls the same `.so` from **C, C++,
Zig, and Python** and shows they agree.

## The demos

```
python -m demos.demo_simple            # minimal terminal run
python -m demos.demo_arm_trajectory    # arm follows an end-effector trajectory (+ viewer)
python -m demos.demo_base_tracking     # base tracks a path with LQR/MPC + observer
python -m demos.demo_pick_and_place    # open-loop (phase-list) pick, drive, place
```

### Compiled control

The same robot, but the controller+estimator are compiled to a Zig kernel and the
host drives the loop from the `.so`. Build a kernel, then run its demo:

```bash
# scenario TOML → lib<name>.so + graph manifest (ReleaseFast)
shinro build scenarios/base_tracking.toml      --import shinro_demo_lekiwi --out build/compiled_base   # base · MPC
shinro build scenarios/base_tracking_mppi.toml --import shinro_demo_lekiwi --out build/compiled_mppi   # base · MPPI
shinro build scenarios/arm_tracking.toml       --import shinro_demo_lekiwi --out build/compiled_arm    # arm  · PID
shinro build scenarios/pickplace_base.toml     --import shinro_demo_lekiwi --out build/compiled_ppbase # base · MPC
shinro build scenarios/pickplace_arm.toml      --import shinro_demo_lekiwi --out build/compiled_pparm  # arm  · PID

python -m demos.demo_compiled_control          # base tracking: [mpc|mppi|all]
python -m demos.demo_compiled_arm              # arm tracking (base still)
python -m demos.demo_compiled_pickplace        # MPC base + PID arm: pick → drive → drop → return
```

Each GIF is a **1800×720 composite** so the tracking is obvious at a glance:

- a 3-D MuJoCo view (top-down for the base, perspective for the arm),
- a **bird's-eye x-y plot** of the reference curve vs the base's actual path
  (or a **3-D end-effector path plot** for the arm),
- a side panel of **tracking error vs time** (position and heading).

**Base tracking (`mpc` / `mppi`)** — the base follows a smooth **B-spline**
reference. The two kernels are a nice contrast: MPC tracks tightly and is
deterministic; MPPI is a sampling controller, visibly looser, and its `epsilon`
perturbations are drawn by the host (the kernel does no sampling).

**Arm tracking (`demo_compiled_arm`)** — the base stays put; the end-effector
follows a 6-D B-spline `[x, y, z, roll, pitch, yaw]`.

**Pick-and-place (`demo_compiled_pickplace`)** — the first **multi-plant** demo:
a compiled **MPC** drives the base around an out-and-back arc while a compiled
**PID** arm reaches down, picks the block, carries it, sets it down, and the base
returns. Two kernels run on one shared sim via `host.drive_channels`, with a
host-side phase machine handling the jaw and the kinematic block. (The arm's
end-effector reference is a *world* pose, so it is offset by the base trajectory
each tick so the arm moves with the base.)

> **On parity and MPPI.** The compiled kernel and the live Python loop are two
> implementations of the same law. For MPC and PID they agree with the whole run.
> MPPI's softmax weighting is *chaotic*: a 1e-16 per-tick difference amplifies to
> O(1) within ~75 ticks even though every tick matches. So the demos compare
> **per tick in lockstep**, not across two independent runs.

## How compiled control works

shinro traces the estimator + controller into a dataflow graph, lowers it, and
compiles it with Zig to a shared object exposing **one** function:

```c
void shinro_step(const double* in, double* out, double* state);
```

It is a **pure function of its ports** — no plant, no RNG, no control flow, no
I/O. The graph manifest (`graph_data_manifest.json`) next to the `.so` lists the
ports (`inputs` / `outputs` / `state_outputs`). Each control tick the host:

```
    sensor ─► [measurement] ┐
            [reference x_ref]├─► pack ports ─► shinro_step(.so) ─► u ─► plant.step(u) ─► sim.step()
            [previous u_prev]┘          ▲                              │
                                          └── feed state_* outputs back ──┘
```

The plant and simulation stay in MuJoCo (`RobotSim`); only the control law is
compiled. Two ports are worth knowing:

- **`state_*` is recurrent** — each tick its output is fed back to the matching
  input. `state_P` (the Kalman covariance) is *host-seeded* to `0.1·I` on the
  first tick, or the kernel starts from the wrong gain.
- **MPPI's `epsilon`** is a free input port — the host draws the perturbations so
  the kernel stays deterministic.

The host loop lives in `src/shinro_demo_lekiwi/host.py` (`drive` for one kernel,
`drive_channels` for several on one sim).

## Cross-language C-ABI interop

Because the kernel is just a C-ABI function, any language can drive it. `interop/`
calls the **same** `libbase.so` from four languages with identical inputs:

```
one shinro_step through four languages — same .so, same inputs:
  C        0.5 -0.5 -0.999999999999999
  C++      0.5 -0.5 -0.999999999999999
  Zig      0.5 -0.5 -0.999999999999999
  Python   0.5 -0.5 -0.999999999999999
```

```bash
make interop        # builds + runs the C, C++, Zig, and Python examples
```

See [`interop/README.md`](./interop/README.md) for the ABI and port layout.

## Install

Python ≥ 3.12 and a **shinro source checkout** (this repo tracks the working
tree, not a published release). Clone both next to each other:

```
parent/
  shinro-python-modules/
  shinro-demo-lekiwi/
```

```bash
cd shinro-demo-lekiwi
make install          # pip install -e ../shinro-python-modules + this repo [mujoco,media,dev]
```

Override the framework path with `make install SHINRO=/path/to/shinro-python-modules`.
Compiling kernels also needs `zig` on `PATH`.

> The bundled MuJoCo model and meshes under `src/shinro_demo_lekiwi/assets/` are
> the LeKiwi / SO-ARM100 assets (see `assets/lekiwi-sim/README.md` for provenance).

## Layout

```
src/shinro_demo_lekiwi/   package: asset paths, helpers, the "lekiwi" preset,
                          host.py (compiled-kernel driver), viz.py, scene.py
  assets/lekiwi-sim/      bundled MJCF + meshes (installed as package data)
configs/                  robot_config.toml + controller/estimator/plant/trajectory TOMLs
scenarios/                complete closed-loop scenario TOMLs (the build spec)
demos/                    runnable demos (`python -m demos.demo_*`)
interop/                  C / C++ / Zig / Python drivers of the compiled kernel
tests/                    full-loop MuJoCo integration suite
```

Scenarios are **plant-authoritative**: `[plant]` names a `type` + `config`, and the
build derives `n_x` / `n_u` from it, so the TOMLs don't repeat the dimensions.

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
