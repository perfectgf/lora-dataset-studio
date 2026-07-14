"""Paste-safe redaction of local home-dir paths.

Shared by anything that emits text meant to be pasted into a PUBLIC GitHub
issue / Discord thread (the diagnostic payload, the per-run "Share
configuration" file): a raw `C:\\Users\\<realname>\\...` (or
`/home/<realname>/...`) path leaks the OS account / Unix username. Only the
drive+Users+<segment> (or /home|Users/<segment>) prefix is swapped for `~`;
the rest of the path is kept, it carries no identity.
"""
import re

# Windows home dir, single OR double backslash (some exception reprs escape
# them): `C:\Users\<name>\...` / `C:\\Users\\<name>\\...`. Case-insensitive
# (drive letter, "users").
_WIN_HOME_RE = re.compile(r'[A-Za-z]:\\{1,2}Users\\{1,2}[^\\/:*?"<>|\r\n]+', re.IGNORECASE)
# POSIX home dir: `/home/<name>` or `/Users/<name>` (macOS).
_POSIX_HOME_RE = re.compile(r'/(?:home|Users)/[^/\r\n]+', re.IGNORECASE)


def redact_user_paths(line):
    """Strip the OS account name out of an absolute home-dir path in a string.
    Only the drive+Users+<segment> (or /home|Users/<segment>) prefix becomes
    `~`; the rest of the path is preserved. NULL/empty passes through."""
    if not line:
        return line
    line = _WIN_HOME_RE.sub('~', line)
    line = _POSIX_HOME_RE.sub('~', line)
    return line
