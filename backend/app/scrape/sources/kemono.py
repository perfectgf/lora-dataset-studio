from ..validators import Platform
from .base import Capabilities
from .gdl_source import GalleryDlSource
from . import registry


class KemonoSource(GalleryDlSource):
    name = 'kemono'
    priority = 100
    category = 'image'   # contenu créateur (souvent photo) → ouvert aux non-admins
    platform_enum = Platform.KEMONO
    cookies_key = 'kemono'   # DDoS-Guard
    capabilities = Capabilities(can_enumerate_profile=True, needs_auth=True,
                                media_kinds=frozenset({'video', 'image'}), own_downloader=True)


registry.register(KemonoSource())
