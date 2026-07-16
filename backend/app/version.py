"""Single source of truth for the app version.

Date-based (YYYY.MM.DD[.N]) so string comparison IS version comparison — the
update check just compares the latest GitHub release tag (stripped of a leading
'v') against this. Bump it when cutting a release ZIP; the Windows source bundle
picks it up automatically (backend/ is copied verbatim into the archive).
"""
APP_VERSION = '2026.07.16.1'
