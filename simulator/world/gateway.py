"""Per-track lifecycle for the persistent mock-world HTTP servers.

The benchmark's central reliability fix (see the plan: *Persistent MCP gateway*):
instead of Hermes spawning the three mock-world servers as fresh stdio subprocesses
on **every** ``hermes -z`` call — a boot that races Hermes's ~0.75s tool-discovery
wait and loses for fast API models — the gateway starts the three servers **once
per track** as long-lived HTTP servers and registers them with Hermes by URL.
Discovery then becomes a connect-to-a-running-server (sub-second, deterministic),
not a per-invocation race.

``WorldGateway`` is a context manager: it picks free loopback ports, spawns the
servers bound to the track's ``world.db`` and clock file, polls each ``/mcp``
endpoint until ready, and guarantees teardown on success, exception, and
``KeyboardInterrupt``. It is injected into the runner the same way ``Harness`` is,
so the fast test suite can substitute a fake and never spawn real servers.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable

from ._server_common import (
    CLOCK_FILE_ENV,
    DEFAULT_HOST,
    HOST_ENV,
    PORT_ENV,
    write_clock,
)
from .registration import WORLD_SERVERS

DEFAULT_READINESS_TIMEOUT = 30.0

# (n) -> n distinct free ports. Injectable so tests can force specific/occupied ports.
PortPicker = Callable[[int], "list[int]"]


@runtime_checkable
class Gateway(Protocol):
    """The lifecycle surface the runner drives — satisfied by ``WorldGateway`` and
    by the test/fake gateways (a no-op gateway, a spy)."""

    urls: dict[str, str]

    def start(self) -> dict[str, str]: ...
    def set_clock(self, when: str) -> None: ...
    def stop(self) -> None: ...


# (world_db, clock_file) -> a started-on-enter gateway. Injected into the runner.
GatewayFactory = Callable[[Path, Path], Gateway]


class GatewayError(RuntimeError):
    """A world server failed to become ready in time — the startup race made loud.

    Raised (with teardown of any already-started servers) instead of letting the
    runner proceed against a half-up gateway, which would resurrect the very
    tool-less-first-turn failure this module exists to remove.
    """


def free_ports(n: int, host: str = DEFAULT_HOST) -> list[int]:
    """Pick ``n`` distinct free loopback ports.

    Binds ``n`` sockets to port 0 *simultaneously* so the OS hands out distinct
    ports (and never an occupied one), then closes them and returns the numbers.
    A small TOCTOU window remains between close and the server's re-bind; the
    readiness poll is what actually confirms each server came up.
    """
    socks: list[socket.socket] = []
    try:
        for _ in range(n):
            s = socket.socket()
            socks.append(s)  # own it before bind() so the finally always closes it
            s.bind((host, 0))
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


class WorldGateway:
    """Start/poll/stop the three per-track mock-world HTTP servers as one unit."""

    def __init__(
        self,
        world_db: str | os.PathLike[str],
        clock_file: str | os.PathLike[str],
        *,
        python_exe: Optional[str] = None,
        host: str = DEFAULT_HOST,
        readiness_timeout: float = DEFAULT_READINESS_TIMEOUT,
        servers: Optional[dict[str, str]] = None,
        port_picker: Optional[PortPicker] = None,
    ) -> None:
        self.world_db = Path(world_db)
        self.clock_file = Path(clock_file)
        self.python_exe = python_exe or sys.executable
        self.host = host
        self.readiness_timeout = readiness_timeout
        self.servers = dict(servers if servers is not None else WORLD_SERVERS)
        self._port_picker = port_picker or (lambda n: free_ports(n, self.host))
        self.urls: dict[str, str] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._started = False

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> dict[str, str]:
        """Spawn all servers, wait until each is reachable, return name→URL.

        On any readiness failure, every already-started server is torn down before
        a :class:`GatewayError` propagates, so a failed start leaves no orphans.
        """
        if self._started:
            return self.urls
        ports = self._port_picker(len(self.servers))
        try:
            for (name, module), port in zip(self.servers.items(), ports):
                self.urls[name] = self._spawn(name, module, port)
            self._await_ready()
        except BaseException:
            self.stop()
            raise
        self._started = True
        return self.urls

    def _spawn(self, name: str, module: str, port: int) -> str:
        env = dict(os.environ)
        env[HOST_ENV] = self.host
        env[PORT_ENV] = str(port)
        env[CLOCK_FILE_ENV] = str(self.clock_file)
        proc = subprocess.Popen(
            [self.python_exe, "-m", module, str(self.world_db)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._procs[name] = proc
        return f"http://{self.host}:{port}/mcp"

    def _await_ready(self) -> None:
        """Poll all servers together until each ``/mcp`` answers, within the timeout.

        Interleaved (not one-server-at-a-time) so a server that crashes on launch
        is detected promptly rather than after waiting out an earlier server's
        full timeout.
        """
        deadline = time.monotonic() + self.readiness_timeout
        pending = set(self.servers)
        while pending and time.monotonic() < deadline:
            for name in list(pending):
                proc = self._procs[name]
                if proc.poll() is not None:
                    raise GatewayError(
                        f"world server {name!r} exited (code {proc.returncode}) "
                        f"before becoming ready at {self.urls[name]}"
                    )
                if _endpoint_ready(self.urls[name]):
                    pending.discard(name)
            if pending:
                time.sleep(0.1)
        if pending:
            raise GatewayError(
                f"world servers {sorted(pending)} not ready within "
                f"{self.readiness_timeout:.0f}s"
            )

    def set_clock(self, when: str) -> None:
        """Stamp the per-track clock file so the servers reflect today's sim time."""
        write_clock(self.clock_file, when)

    def stop(self) -> None:
        """Terminate and reap every server. Idempotent; safe to call repeatedly."""
        # Hand off the registry first so a teardown error can't leave a partially
        # reaped set behind for a second stop() / __exit__ to re-terminate.
        procs, self._procs = self._procs, {}
        self.urls = {}  # drop URLs too, so a re-start can't register stale endpoints
        self._started = False
        for proc in procs.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in procs.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                # Bounded wait even after SIGKILL: a child wedged in uninterruptible
                # sleep (D-state) can't be reaped, and we must not hang teardown (and
                # thus the whole run) on it. Give up after the timeout — at worst one
                # orphaned process, never a stalled benchmark.
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass

    # --- context manager (teardown on success, exception, KeyboardInterrupt) --

    def __enter__(self) -> "WorldGateway":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False  # never suppress; teardown still ran for exceptions/Ctrl-C


def _endpoint_ready(url: str) -> bool:
    """True once the ``/mcp`` endpoint answers at all (even 4xx => server is up)."""
    try:
        urllib.request.urlopen(url, timeout=1)
        return True
    except urllib.error.HTTPError:
        return True  # e.g. 406 Not Acceptable for a plain GET — it's serving
    except (urllib.error.URLError, OSError):
        return False
