// Zig-ABI interop: dlopen the compiled LeKiwi kernel and call shinro_step once.
// Same port layout as the C example (see interop/c/kernel_demo.c).
//
// The .so path comes from the LIBBASE_SO env var (set by interop/Makefile),
// defaulting to ../build/compiled_base/lib/libbase.so. Output goes through C's
// printf so the formatting matches the other languages exactly.
//
// Build:  zig build-exe -O ReleaseFast -lc -femit-bin=kernel_demo_zig kernel_demo.zig
const std = @import("std");

const N_IN = 21;
const N_OUT = 3;
const N_STATE = 15;

const ShinroStep = *const fn ([*]const f64, [*]f64, [*]f64) callconv(.c) void;

extern fn printf(fmt: [*:0]const u8, ...) c_int;
extern fn getenv(name: [*:0]const u8) ?[*:0]u8;

pub fn main() !void {
    const path = if (getenv("LIBBASE_SO")) |p| std.mem.span(p) else "../build/compiled_base/lib/libbase.so";

    var lib = try std.DynLib.open(path);
    defer lib.close();
    const step = lib.lookup(ShinroStep, "shinro_step") orelse return error.SymbolNotFound;

    var in: [N_IN]f64 = [_]f64{0} ** N_IN;
    var out: [N_OUT]f64 = [_]f64{0} ** N_OUT;
    var state: [N_STATE]f64 = [_]f64{0} ** N_STATE;
    in[0] = 0.1;
    in[1] = 0.2;
    in[2] = 0.3;
    in[3] = 0.5;
    in[4] = 0.0;
    in[5] = 0.0;
    in[12] = 0.1;
    in[16] = 0.1;
    in[20] = 0.1;

    step(&in, &out, &state);
    _ = printf("%.15g %.15g %.15g\n", out[0], out[1], out[2]);
}
