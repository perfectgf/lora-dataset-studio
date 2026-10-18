"""Dependency-free helpers used while starting the LDS web server."""
import os
import socket
import sys

# Linux keeps the connections a closed server accepted in TIME_WAIT for about a
# minute. Werkzeug binds with SO_REUSEADDR, which Linux accepts over TIME_WAIT
# but never over a live listener; a probe without it called the port taken
# right after every restart, and the startup slide moved LDS off the port its
# container publishes (2026-10-18, Docker GPU: "Apply and restart" came back
# on 5051 while Docker publishes 5050 only). Linux only: Windows probes with
# SO_EXCLUSIVEADDRUSE below, and BSD lets SO_REUSEADDR bind a wildcard over a
# specific listener, which would hide the collision _wildcard_is_free reports.
_PROBE_REUSES_TIME_WAIT = sys.platform.startswith("linux")
_TRUTHY = {"1", "true", "yes", "on"}


def _prepare_probe(probe: socket.socket) -> None:
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    elif _PROBE_REUSES_TIME_WAIT:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


def _wildcard_is_free(port: int) -> bool:
    """Is the port free for EVERY address, not just the one we were asked about?

    A listener on the wildcard address (0.0.0.0 / ::) collides with a bind to any
    specific address on the same port — but Windows does not report it that way.
    It lets the more specific bind succeed, because 127.0.0.1 is a closer match
    than 0.0.0.0. So probing only the configured host answers "free" while the
    port is very much taken, and the second server starts anyway: two processes
    on one port, the specific one capturing loopback traffic, no error at all.

    That is the real collision LDS hits — a second launch while the first is
    still running in LAN mode, which binds the wildcard.

    A family we cannot even create a socket for is not holding anything, so it is
    skipped rather than counted as occupied; treating an IPv6-less host as "every
    port is taken" would push the search through all its attempts and then fail.
    """
    for family, address in ((socket.AF_INET, ("0.0.0.0", port)),
                            (socket.AF_INET6, ("::", port))):
        try:
            probe = socket.socket(family, socket.SOCK_STREAM)
        except OSError:
            continue
        with probe:
            _prepare_probe(probe)
            try:
                probe.bind(address)
            except OSError:
                return False
    return True


def can_bind(host: str, port: int) -> bool:
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM,
                                       flags=socket.AI_PASSIVE)
    except socket.gaierror:
        return False
    bound_any = False
    for family, socktype, proto, _, sockaddr in addresses:
        try:
            with socket.socket(family, socktype, proto) as candidate:
                _prepare_probe(candidate)
                candidate.bind(sockaddr)
            bound_any = True
        except OSError:
            # A hostname such as localhost can resolve to IPv4 and IPv6. If
            # either address is occupied, Werkzeug may choose it and fail.
            return False
    # Binding the requested address is necessary but NOT sufficient: a wildcard
    # listener makes that bind succeed while the port is taken. See above.
    return bound_any and _wildcard_is_free(port)


def find_available_port(host: str, preferred_port: int, max_attempts: int = 100) -> int:
    if not 1 <= preferred_port <= 65535:
        raise ValueError(f"Invalid server port: {preferred_port}")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    stop = min(65536, preferred_port + max_attempts)
    for port in range(preferred_port, stop):
        if can_bind(host, port):
            return port
    raise OSError(f"No available TCP port found from {preferred_port} to {stop - 1}")


def startup_port(host: str, requested_port: int, environ=None) -> int:
    """The port the server binds at startup.

    It slides past a busy port only when the bind belongs to the app. A
    container publishes one fixed port (LDS_BIND_MANAGED, set by both Docker
    images), and LDS_AUTO_PORT=0 asks for the same: there a slide leaves LDS
    serving a port nothing publishes. Werkzeug then binds the requested port
    itself, and a port that is really held fails the start, which the
    container supervisor retries."""
    environ = os.environ if environ is None else environ
    if (environ.get("LDS_AUTO_PORT") == "0"
            or environ.get("LDS_BIND_MANAGED", "").strip().lower() in _TRUTHY):
        return requested_port
    return find_available_port(host, requested_port)
