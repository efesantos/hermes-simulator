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
from typing import Callable, Optional

from ._server_common import write_clock
from .registration import WORLD_SERVERS

DEFAULT_HOST = "127.0.0.1"
DEFAULT_READINESS_TIMEOUT = 30.0

# (n) -> n distinct free ports. Injectable so tests can force specific/occupied ports.
PortPicker = Callable[[int], "list[int]"]
# (world_db, clock_file) -> a started-on-enter gateway. Injected into the runner.
GatewayFactory = Callable[[Path, Path], "WorldGateway"]


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
            s.bind((host, 0))
            socks.append(s)
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
        self._procs: list[subprocess.Popen] = []
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
            for name in self.servers:
                self._await_ready(name)
        except BaseException:
            self.stop()
            raise
        self._started = True
        return self.urls

    def _spawn(self, name: str, module: str, port: int) -> str:
        env = dict(os.environ)
        env["HERMES_MCP_HOST"] = self.host
        env["HERMES_MCP_PORT"] = str(port)
        env["HERMES_SIM_NOW_FILE"] = str(self.clock_file)
        proc = subprocess.Popen(
            [self.python_exe, "-m", module, str(self.world_db)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._procs.append(proc)
        return f"http://{self.host}:{port}/mcp"

    def _await_ready(self, name: str) -> None:
        url = self.urls[name]
        proc = self._procs[list(self.servers).index(name)]
        deadline = time.monotonic() + self.readiness_timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise GatewayError(
                    f"world server {name!r} exited (code {proc.returncode}) before "
                    f"becoming ready at {url}"
                )
            if _endpoint_ready(url):
                return
            time.sleep(0.1)
        raise GatewayError(
            f"world server {name!r} not ready at {url} within "
            f"{self.readiness_timeout:.0f}s"
        )

    def set_clock(self, when: str) -> None:
        """Stamp the per-track clock file so the servers reflect today's sim time."""
        write_clock(self.clock_file, when)

    def stop(self) -> None:
        """Terminate and reap every server. Idempotent; safe to call repeatedly."""
        for proc in self._procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in self._procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        self._procs = []
        self._started = False

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


def default_gateway_factory(world_db: Path, clock_file: Path) -> WorldGateway:
    """The real factory the runner uses; tests inject a fake of the same shape."""
    return WorldGateway(world_db, clock_file)
