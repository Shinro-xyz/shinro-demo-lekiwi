# Cross-language C-ABI interop

The compiled control kernel is a language-agnostic C-ABI artifact:

```c
void shinro_step(const double* in, double* out, double* state);
```

Nothing about it is Python- or shinro-specific — any language that can `dlopen`
a shared library can drive it. This directory calls the **same** `.so` through
C, C++, Zig, and Python with identical inputs and shows they agree.

Build the artifact first (from the repo root):

```bash
shinro build scenarios/base_tracking.toml --import shinro_demo_lekiwi --out build/compiled_base
```

Then:

```bash
make -C interop        # or: make interop   (from the repo root)
```

```
one shinro_step through four languages — same .so, same inputs:
  C        0.5 -0.5 -1
  C++      0.5 -0.5 -1
  Zig      0.5 -0.5 -1
  Python   0.5 -0.5 -1
```

## Port layout

From `build/compiled_base/graph_data_manifest.json` (for the base-tracking
KF+LQR graph; other graphs differ):

| Port  | Elements | Layout |
| ----- | -------- | ------ |
| in    | 21 | `y(3) │ x_ref(3) │ u_prev(3) │ state_x_hat(3) │ state_P(9)` |
| out   | 3  | `u` — the control |
| state | 15 | `state_x_hat(3) │ state_P(9) │ state_u_prev(3)` |

The `state_*` ports are recurrent: feed each one back to its matching input on
the next tick. `state_P` (the Kalman-filter covariance) is **host-seeded** —
initialise it to `0.1*I` on the first tick or the kernel uses the wrong gain.
`u_prev` is the previously applied `u`.

## Files

| Language | File | Build |
| -------- | ---- | ----- |
| C   | `c/kernel_demo.c`       | `cc -O2 -o kernel_demo_c kernel_demo.c -ldl` |
| C++ | `cpp/kernel_demo.cpp`   | `c++ -std=c++17 -O2 -o kernel_demo_cpp kernel_demo.cpp -ldl` |
| Zig | `zig/kernel_demo.zig`   | `zig build-exe -O ReleaseFast -lc -femit-bin=kernel_demo_zig kernel_demo.zig` |
| Python | `python/kernel_demo.py` | `python3 python/kernel_demo.py <so>` |
