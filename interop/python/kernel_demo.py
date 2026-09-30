"""Python-ABI interop: ctypes load of the compiled LeKiwi kernel.

Same port layout as the C example (see interop/c/kernel_demo.c).

Run:  python3 kernel_demo.py /path/to/libbase.so
"""

import ctypes
import sys

N_IN, N_OUT, N_STATE = 21, 3, 15


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "../build/compiled_base/lib/libbase.so"
    lib = ctypes.CDLL(path)
    ptr = ctypes.POINTER(ctypes.c_double)
    lib.shinro_step.argtypes = [ptr, ptr, ptr]
    lib.shinro_step.restype = None

    inp = (ctypes.c_double * N_IN)()
    out = (ctypes.c_double * N_OUT)()
    state = (ctypes.c_double * N_STATE)()
    inp[0], inp[1], inp[2] = 0.1, 0.2, 0.3  # y
    inp[3], inp[4], inp[5] = 0.5, 0.0, 0.0  # x_ref
    inp[12] = inp[16] = inp[20] = 0.1       # state_P = 0.1*I

    lib.shinro_step(inp, out, state)
    print(f"{out[0]:.15g} {out[1]:.15g} {out[2]:.15g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
