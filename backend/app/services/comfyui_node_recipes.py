"""Reviewed node recipes for the core's own Setup actions.

Plugin recipes belong to their own manifests. Each recipe here pins one commit
archive by its SHA-256: comfyui_node_install prepares it into custom_nodes on
any ComfyUI layout, and adds its Python requirements only on the portable layout
whose Python LDS manages. Elsewhere a recipe with requirements returns the exact
steps instead (see the node-pack rule in setup_installer).
"""
from ..plugins.node_recipe import NodeRecipe

# Krea 2 Identity Edit nodes, v1.2.5 (Apache-2.0): the version LDS validated the
# Krea 2 Edit engine against. The pyproject declares `dependencies = []`, so the
# pack is only files. A newer upstream release is adopted here deliberately,
# after its graph has been checked, never by following a moving branch.
KREA = NodeRecipe(
    id='comfyui-krea2edit', version='1.2.5', folder='comfyui-krea2edit',
    url='https://lds.perfectgf.fr/models/tools/ff1102a1d17f597d8b100afd17bd489a7910325729483398f1385c541f18c171/comfyui-krea2edit.zip',
    sha256='ff1102a1d17f597d8b100afd17bd489a7910325729483398f1385c541f18c171',
    archive_prefix='comfyui-krea2edit-86f886dac23013d88996e3a2e99093ba44d322fb',
    expected_classes=('Krea2EditModelPatch', 'Krea2EditGroundedEncode'),
)

# Detail Daemon sampler, v1.2.0 (MIT): node 57 of the SDXL Test Studio graph.
# detail_daemon_node.py imports matplotlib at module level and the pack's
# requirements.txt names it without a version. 3.11.2 is the release the SeedVR2
# recipe pins too, so the two never ask one ComfyUI Python for different copies.
# PyPI publishes its wheels for CPython 3.11 to 3.15, the recipe's Python range,
# so a portable on another Python is told so before anything is downloaded. Its
# own dependencies (contourpy, cycler, fonttools, kiwisolver, pyparsing,
# python-dateutil, and numpy, pillow, packaging that ComfyUI already has)
# include no GPU package.
DETAIL_DAEMON = NodeRecipe(
    id='comfyui-detail-daemon', version='1.2.0', folder='ComfyUI-Detail-Daemon',
    url='https://lds.perfectgf.fr/models/tools/5f0eaa7774d5d9d3daccdb5e77b97f416dfadcfcf52ecb14205ce3c0784d6c6d/ComfyUI-Detail-Daemon.zip',
    sha256='5f0eaa7774d5d9d3daccdb5e77b97f416dfadcfcf52ecb14205ce3c0784d6c6d',
    archive_prefix='ComfyUI-Detail-Daemon-3394e44afea04ed0188fb37b21f0d9952469766b',
    expected_classes=('DetailDaemonSamplerNode',),
    requirements=('matplotlib==3.11.2',),
    python='>=3.11,<3.16',
)

CORE_RECIPES = {'krea_nodes': KREA, 'detail_daemon_nodes': DETAIL_DAEMON}
