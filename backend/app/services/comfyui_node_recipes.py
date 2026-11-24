"""Reviewed node recipes used by core image preparation.

Plugin recipes belong to their own manifests. This small catalogue only covers
the core's existing Krea action; it never installs another image/video engine.
"""
from ..plugins.node_recipe import NodeRecipe

KREA = NodeRecipe(
    id='comfyui-krea2edit', version='1.2.5', folder='comfyui-krea2edit',
    url='https://lds.perfectgf.fr/models/tools/ff1102a1d17f597d8b100afd17bd489a7910325729483398f1385c541f18c171/comfyui-krea2edit.zip',
    sha256='ff1102a1d17f597d8b100afd17bd489a7910325729483398f1385c541f18c171',
    archive_prefix='comfyui-krea2edit-86f886dac23013d88996e3a2e99093ba44d322fb',
    expected_classes=('Krea2EditModelPatch', 'Krea2EditGroundedEncode'),
)

CORE_RECIPES = {'krea_nodes': KREA}
