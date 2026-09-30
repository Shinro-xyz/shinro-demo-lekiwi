"""LeKiwi reference-robot demo for the shinro control framework.

Importing this package registers its ``[physics].preset = "lekiwi"`` with
shinro's physics-preset registry, so scenarios that declare the preset resolve
without an extra ``--import``. It also re-exports the asset paths and the
simulation helpers the demos and tests share.
"""

from shinro_demo_lekiwi import presets as _presets  # noqa: F401  (registers "lekiwi" on import)
from shinro_demo_lekiwi.helpers import inject_free_joint, load_lekiwi_mjcf, load_model_assets
from shinro_demo_lekiwi.lekiwi_sim import LeKiwiSim
from shinro_demo_lekiwi.paths import ASSETS, HERE, MESH_DIR, MJCF_PATH
from shinro_demo_lekiwi.scene import inject_graspable_block

__all__ = [
    "ASSETS",
    "HERE",
    "MJCF_PATH",
    "MESH_DIR",
    "LeKiwiSim",
    "inject_free_joint",
    "inject_graspable_block",
    "load_lekiwi_mjcf",
    "load_model_assets",
]
