// C++-ABI interop: dlopen the compiled LeKiwi kernel and call shinro_step once.
// Same port layout as the C example (see interop/c/kernel_demo.c).
//
// Build:  c++ -std=c++17 -O2 -o kernel_demo_cpp kernel_demo.cpp -ldl
// Run:    ./kernel_demo_cpp /path/to/libbase.so
#include <dlfcn.h>

#include <array>
#include <cstdio>

namespace {
constexpr int N_IN = 21;
constexpr int N_OUT = 3;
constexpr int N_STATE = 15;
using ShinroStep = void (*)(const double *, double *, double *);
}  // namespace

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../build/compiled_base/lib/libbase.so";

    void *lib = dlopen(path, RTLD_NOW);
    if (!lib) {
        std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }
    auto step = reinterpret_cast<ShinroStep>(dlsym(lib, "shinro_step"));
    if (!step) {
        std::fprintf(stderr, "dlsym failed: %s\n", dlerror());
        return 1;
    }

    std::array<double, N_IN> in{};
    std::array<double, N_OUT> out{};
    std::array<double, N_STATE> state{};
    in[0] = 0.1; in[1] = 0.2; in[2] = 0.3;        // y
    in[3] = 0.5; in[4] = 0.0; in[5] = 0.0;        // x_ref
    in[12] = 0.1; in[16] = 0.1; in[20] = 0.1;     // state_P = 0.1*I

    step(in.data(), out.data(), state.data());
    std::printf("%.15g %.15g %.15g\n", out[0], out[1], out[2]);

    dlclose(lib);
    return 0;
}
