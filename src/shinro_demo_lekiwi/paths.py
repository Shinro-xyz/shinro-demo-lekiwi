"""Filesystem locations of the bundled LeKiwi MuJoCo assets.

``HERE`` is this package directory, so the paths work both from a source
checkout and from an installed wheel (the ``assets/`` tree is package data).
"""

from pathlib import Path

HERE = Path(__file__).parent
ASSETS = HERE / "assets" / "lekiwi-sim"
MJCF_PATH = str(ASSETS / "mjcf_lcmm_robot.xml")
MESH_DIR = ASSETS / "meshes"
