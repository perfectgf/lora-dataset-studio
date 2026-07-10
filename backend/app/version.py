"""Single source of truth for the app version.

Date-based (YYYY.MM.DD[.N]) so string comparison IS version comparison — the
update check just compares the latest GitHub release tag (stripped of a leading
'v') against this. Bump it when cutting a release zip; the portable bundle picks
it up automatically (backend/ is copied verbatim into the bundle).
"""
APP_VERSION = '2026.07.10'
