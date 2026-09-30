.PHONY: install test test-quick demo

# Path to the shinro framework checkout (sibling by default).
SHINRO ?= ../shinro-python-modules

# Install the framework checkout + this repo's extras.
install:
	pip install -e "$(SHINRO)"
	pip install -e ".[mujoco,media,dev]"

# Full-loop MuJoCo integration suite.
test:
	python3 -m pytest tests/ -v --tb=short

# Fast import/preset smoke check (no simulations).
test-quick:
	python3 -c "import shinro_demo_lekiwi as d; print('preset registered:', d.MJCF_PATH)"

demo:
	python3 -m demos.demo_simple
