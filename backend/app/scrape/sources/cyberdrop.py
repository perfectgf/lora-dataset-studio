from ..validators import Platform
from .base import Capabilities
from .gdl_source import GalleryDlSource
from . import registry


class CyberdropSource(GalleryDlSource):
    name = 'cyberdrop'
    priority = 100
    category = 'image'   # hébergeur de fichiers (souvent photo) → ouvert aux non-admins
    platform_enum = Platform.CYBERDROP
    gdl_opts = None
    capabilities = Capabilities(can_enumerate_profile=True, polite=True,
                                media_kinds=frozenset({'video', 'image'}), own_downloader=True)


registry.register(CyberdropSource())
