# app/scrape/sources/registry.py
"""Registry de sources : résolution d'une URL vers LA source qui la gère.

Remplace le triple-dispatch (validators / routes / download_service). Les sources
sont triées par priorité décroissante ; resolve(url) renvoie le 1er Match truthy,
donc une source dédiée (priorité 100) gagne sur la fallback universelle (0)."""
import logging
from typing import Optional

from .base import Source, Match


logger = logging.getLogger(__name__)


class _Registry:
    def __init__(self):
        self._sources = []  # type: list[Source]

    def register(self, src: Source) -> None:
        self._sources.append(src)

    def all_sources(self) -> list:
        return sorted(self._sources, key=lambda s: s.priority, reverse=True)

    def resolve(self, url: str) -> Optional[Match]:
        for src in self.all_sources():
            try:
                m = src.match(url)
            except Exception as exc:
                logger.warning("Source %r match() a levé, ignorée: %r",
                               getattr(src, 'name', '?'), exc)
                m = None
            if m is not None:
                m.source = src
                return m
        return None

    def assert_one_universal(self) -> None:
        names = [s.name for s in self._sources]
        if len(names) != len(set(names)):
            raise RuntimeError(f"Sources de scraping : noms en double ({names})")
        universals = [s.name for s in self._sources
                      if getattr(s.capabilities, 'is_universal_fallback', False)]
        if len(universals) != 1:
            raise RuntimeError(
                f"Il doit y avoir EXACTEMENT une source universelle, trouvé : {universals}")


# Registry global du module (peuplé par les imports dans sources/__init__.py).
_registry = _Registry()


def register(src: Source) -> None:
    _registry.register(src)


def resolve(url: str) -> Optional[Match]:
    return _registry.resolve(url)


def all_sources():
    return _registry.all_sources()


def assert_one_universal() -> None:
    _registry.assert_one_universal()
