/* C-ABI interop: dlopen the compiled LeKiwi kernel and call shinro_step once.
 *
 * The kernel is a pure function of its ports:
 *     void shinro_step(const double* in, double* out, double* state);
 * with the port layout in build/compiled_base/graph_data_manifest.json:
 *     in    = y(3) | x_ref(3) | u_prev(3) | state_x_hat(3) | state_P(3x3)  -> 21
 *     out   = u(3)                                                        ->  3
 *     state = state_x_hat(3) | state_P(9) | state_u_prev(3)               -> 15
 *
 * Build:  cc -O2 -o kernel_demo_c kernel_demo.c -ldl
 * Run:    ./kernel_demo_c /path/to/libbase.so
 */
#include <dlfcn.h>
#include <stdio.h>

#define N_IN 21
#define N_OUT 3
#define N_STATE 15

typedef void (*shinro_step_fn)(const double *, double *, double *);

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../build/compiled_base/lib/libbase.so";

    void *lib = dlopen(path, RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }
    shinro_step_fn step = (shinro_step_fn)dlsym(lib, "shinro_step");
    if (!step) {
        fprintf(stderr, "dlsym failed: %s\n", dlerror());
        return 1;
    }

    double in[N_IN] = {0}, out[N_OUT] = {0}, state[N_STATE] = {0};
    in[0] = 0.1; in[1] = 0.2; in[2] = 0.3;                 /* y          */
    in[3] = 0.5; in[4] = 0.0; in[5] = 0.0;                 /* x_ref      */
    /* in[6..8] u_prev = 0, in[9..11] state_x_hat = 0        */
    in[12] = 0.1; in[16] = 0.1; in[20] = 0.1;              /* state_P = 0.1*I */

    step(in, out, state);
    printf("%.15g %.15g %.15g\n", out[0], out[1], out[2]);

    dlclose(lib);
    return 0;
}
