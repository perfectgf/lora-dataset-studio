from ..validators import Platform
from .base import Capabilities
from .gdl_source import GalleryDlSource
from . import registry


class CoomerSource(GalleryDlSource):
    name = 'coomer'
    priority = 100
    category = 'image'   # contenu créateur (souvent photo) → ouvert aux non-admins
    platform_enum = Platform.COOMER
    cookies_key = 'coomer'   # DDoS-Guard
    capabilities = Capabilities(can_enumerate_profile=True, needs_auth=True,
                                media_kinds=frozenset({'video', 'image'}), own_downloader=True)


registry.register(CoomerSource())
