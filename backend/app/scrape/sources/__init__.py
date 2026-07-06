# app/scrape/sources/__init__.py
"""Scrapers par source (énumération de médias d'un profil / niche / listing).

Chaque module expose `scan(validation_result) -> (items, error)` où items est une
liste de dicts au schéma commun :
    { 'url', 'title', 'thumbnail', 'type' ('video'|'image'), 'platform', ... }
Le téléchargement effectif d'un item passe par /api/scrape/download (yt-dlp ou
stratégie dédiée), pas par ces modules.
"""
# Enregistrement explicite des sources (ordre indifférent : la priorité décide).
# AJOUTER UNE SOURCE = créer sources/<name>.py (sous-classe de base.Source,
# match()/scan()/(optionnel)download() + registry.register(...) en bas de fichier)
# PUIS l'importer ici. Pas de pkgutil (imports silencieux / ordre non déterministe).
from . import registry   # noqa: F401  (expose le registry + crée _registry avant les sources)
from . import redgifs    # noqa: F401
from . import instagram  # noqa: F401
from . import picazor    # noqa: F401
from . import erome      # noqa: F401
from . import coomer     # noqa: F401
from . import kemono     # noqa: F401
from . import bunkr      # noqa: F401
from . import cyberdrop  # noqa: F401
from . import x          # noqa: F401
from . import tiktok     # noqa: F401
from . import image_sites  # noqa: F401  (pornpics — vraies photos par catégorie)
from . import civitai     # noqa: F401  (civitai.com/.red — images IA par tag)
from . import fapello     # noqa: F401  (fapello.com + miroirs de langue — page modèle)
from . import universal  # noqa: F401

# Invariant : exactement une source universelle, noms uniques. Lève au démarrage
# si violé (mieux qu'un bug de dispatch silencieux).
registry.assert_one_universal()
