"""Registered ``[physics].preset`` factory for the LeKiwi model.

A preset is shinro's plugin seam for turning a scenario's ``[physics]`` section
into an engine model. Importing this module registers ``"lekiwi"`` with
``shinro.factories.registry``; :mod:`shinro_demo_lekiwi` imports it eagerly, so
a scenario that declares ``[physics].preset = "lekiwi"`` resolves as soon as the
package is imported (no separate ``--import`` needed for in-repo use).
"""

from pathlib import Path

from shinro.factories.registry import PhysicsModel, register_physics_preset

from shinro_demo_lekiwi.helpers import inject_free_joint, load_model_assets
from shinro_demo_lekiwi.paths import MESH_DIR, MJCF_PATH


@register_physics_preset("lekiwi")
def lekiwi_preset(cfg: dict) -> PhysicsModel:
    """Build the LeKiwi engine model from the bundled MJCF + meshes.

    The stock MJCF has the arm as a sibling of the wheel base; this nests the
    arm under a free-jointed wheel chassis so the base can drive in MuJoCo.

    Args:
        cfg: The scenario's ``[physics]`` section (currently unused; a preset
            may read it to vary the model).

    Returns:
        :class:`~shinro.factories.registry.PhysicsModel` holding the rewritten
        MJCF text and the mesh assets (filename -> bytes).
    """
    xml = inject_free_joint(Path(MJCF_PATH).read_text())
    return PhysicsModel(xml_string=xml, assets=load_model_assets(MESH_DIR))
