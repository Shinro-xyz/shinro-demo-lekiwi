"""Shared fixtures for the LeKiwi full-loop integration suite.

Requires MuJoCo plus the bundled LeKiwi assets. The ``mujoco`` import is delayed
into :func:`mujoco_available` so an environment without the optional extra skips
cleanly at fixture time instead of failing at collection.
"""

from pathlib import Path

import pytest

from shinro_demo_lekiwi import MJCF_PATH, inject_free_joint, load_model_assets
from shinro_demo_lekiwi.paths import MESH_DIR


@pytest.fixture(scope="session")
def mujoco_available():
    """Skip the suite when the optional ``[mujoco]`` extra is not installed."""
    pytest.importorskip("mujoco")
    return True


@pytest.fixture(scope="session")
def lekiwi_xml(mujoco_available):
    """Raw LeKiwi MJCF document as a string."""
    return Path(MJCF_PATH).read_text()


@pytest.fixture(scope="session")
def lekiwi_assets(mujoco_available):
    """Dict of mesh filename -> bytes for ``xml_string`` loading."""
    return load_model_assets(MESH_DIR)


@pytest.fixture(scope="session")
def lekiwi_xml_freejoint(mujoco_available, lekiwi_xml):
    """LeKiwi MJCF with the arm nested under the base and a free joint injected."""
    return inject_free_joint(lekiwi_xml)
