from ..validators import Platform
from .base import Capabilities
from .gdl_source import GalleryDlSource
from . import registry


class TiktokSource(GalleryDlSource):
    name = 'tiktok'
    priority = 100
    platform_enum = Platform.TIKTOK
    cookies_key = 'tiktok'
    capabilities = Capabilities(can_enumerate_profile=True, needs_auth=True,
                                media_kinds=frozenset({'video', 'image'}), own_downloader=True)


registry.register(TiktokSource())
