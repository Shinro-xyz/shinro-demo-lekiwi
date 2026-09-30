# Cross-language C-ABI interop for the compiled LeKiwi kernel.
#
# Build the artifact first (from the repo root):
#   shinro build scenarios/base_tracking.toml --import shinro_demo_lekiwi --out build/compiled_base
#
# Then:   make -C interop          (or `make interop` from the repo root)
#
# The same .so is driven through C, C++, Zig, and Python with identical inputs;
# the four printed control vectors must agree.

SO  ?= ../build/compiled_base/lib/libbase.so
CC  ?= cc
CXX ?= c++
ZIG ?= zig

BIN := build

.PHONY: all run clean

all: run

$(BIN):
	mkdir -p $(BIN)

$(BIN)/kernel_demo_c: c/kernel_demo.c | $(BIN)
	$(CC) -O2 -o $@ $< -ldl

$(BIN)/kernel_demo_cpp: cpp/kernel_demo.cpp | $(BIN)
	$(CXX) -std=c++17 -O2 -o $@ $< -ldl

$(BIN)/kernel_demo_zig: zig/kernel_demo.zig | $(BIN)
	$(ZIG) build-exe -O ReleaseFast -lc -femit-bin=$@ $<

run: $(BIN)/kernel_demo_c $(BIN)/kernel_demo_cpp $(BIN)/kernel_demo_zig
	@test -f "$(SO)" || { echo "no $(SO) — build it from the repo root first:"; \
		echo "  shinro build scenarios/base_tracking.toml --import shinro_demo_lekiwi --out build/compiled_base"; exit 1; }
	@echo "one shinro_step through four languages — same .so, same inputs:"
	@printf "  %-8s %s\n" C      "$$($(BIN)/kernel_demo_c     $(SO))"
	@printf "  %-8s %s\n" C++    "$$($(BIN)/kernel_demo_cpp   $(SO))"
	@printf "  %-8s %s\n" Zig    "$$(LIBBASE_SO=$(SO) $(BIN)/kernel_demo_zig)"
	@printf "  %-8s %s\n" Python "$$(python3 python/kernel_demo.py $(SO))"

clean:
	rm -rf $(BIN)
