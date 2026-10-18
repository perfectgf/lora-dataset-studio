import socket
import sys
import time

import pytest
from port_utils import find_available_port, startup_port


def test_uses_available_preferred_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    assert find_available_port("127.0.0.1", port) == port


def test_advances_past_occupied_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        occupied = listener.getsockname()[1]
        listener.listen()
        assert find_available_port("127.0.0.1", occupied) > occupied


def test_localhost_does_not_ignore_an_occupied_ipv4_address():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        occupied = listener.getsockname()[1]
        listener.listen()
        assert find_available_port("localhost", occupied) > occupied


def test_a_wildcard_listener_makes_the_port_unavailable_for_a_specific_host():
    """The collision that actually happens: LDS is already running in LAN mode,
    so it holds 0.0.0.0:<port>, and the second launch asks for 127.0.0.1:<port>.

    Windows lets a bind to a SPECIFIC address succeed while another socket holds
    the wildcard — the specific address is the more precise match. So the probe
    answers "free", the second server starts on the same port, and two servers
    end up sharing it with the specific one capturing loopback traffic. Nothing
    raises; it simply looks like it worked.

    The question worth asking is "is this PORT free", not "can I bind THIS
    address" — the two only diverge here, which is exactly the case users hit."""
    with socket.socket() as listener:
        listener.bind(("0.0.0.0", 0))
        occupied = listener.getsockname()[1]
        listener.listen()
        assert find_available_port("127.0.0.1", occupied) > occupied


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="TIME_WAIT bind rules of Linux: Docker images and Linux installs")
def test_a_port_left_in_time_wait_by_a_restarted_server_is_free():
    """Right after a restart the previous server's accepted connections stay in
    TIME_WAIT for about a minute. Werkzeug binds over them (SO_REUSEADDR), so
    the probe must agree; otherwise the relaunch slides to the next port, which
    in Docker nothing publishes (2026-10-18, "Apply and restart" on Docker GPU)."""
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.listen()
    client = socket.create_connection(("127.0.0.1", port))
    accepted, _ = server.accept()
    accepted.close()          # the exiting server closes first...
    time.sleep(0.1)
    client.close()            # ...so its side of the connection enters TIME_WAIT
    time.sleep(0.1)
    server.close()
    assert find_available_port("127.0.0.1", port) == port


@pytest.mark.parametrize("environ", [{"LDS_BIND_MANAGED": "1"}, {"LDS_BIND_MANAGED": "true"},
                                     {"LDS_AUTO_PORT": "0"}])
def test_a_managed_bind_never_slides_off_the_requested_port(environ):
    """A container publishes one fixed port: serving the next one leaves LDS
    unreachable behind a healthy-looking supervisor."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        occupied = listener.getsockname()[1]
        listener.listen()
        assert startup_port("127.0.0.1", occupied, environ) == occupied


def test_an_unmanaged_bind_still_slides_past_a_busy_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        occupied = listener.getsockname()[1]
        listener.listen()
        assert startup_port("127.0.0.1", occupied, {}) > occupied


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_rejects_invalid_ports(port):
    with pytest.raises(ValueError, match="Invalid server port"):
        find_available_port("127.0.0.1", port)


def test_rejects_empty_search_range():
    with pytest.raises(ValueError, match="max_attempts"):
        find_available_port("127.0.0.1", 5050, max_attempts=0)
