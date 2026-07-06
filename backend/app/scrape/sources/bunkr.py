from ..validators import Platform
from .base import Capabilities
from .gdl_source import GalleryDlSource
from . import registry


class BunkrSource(GalleryDlSource):
    name = 'bunkr'
    priority = 100
    category = 'image'   # hébergeur de fichiers (souvent photo) → ouvert aux non-admins
    platform_enum = Platform.BUNKR
    gdl_opts = ['-o', 'extractor.bunkr.tlds=true']   # TLDs rotatifs gérés côté gallery-dl
    capabilities = Capabilities(can_enumerate_profile=True, polite=True,
                                media_kinds=frozenset({'video', 'image'}), own_downloader=True)


registry.register(BunkrSource())
