from ..validators import Platform
from .base import Capabilities
from .gdl_source import GalleryDlSource
from . import registry


class XSource(GalleryDlSource):
    name = 'x'
    priority = 100
    platform_enum = Platform.X
    cookies_key = 'x'   # x.com requiert des cookies authentifiés (timelines/NSFW)
    capabilities = Capabilities(can_enumerate_profile=True, needs_auth=True, polite=True,
                                media_kinds=frozenset({'video', 'image'}), own_downloader=True)


registry.register(XSource())
