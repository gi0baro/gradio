"""
Defines helper methods useful for setting up ports, launching servers, and
creating tunnels.
"""
from __future__ import annotations

import contextvars
import os
import socket
import threading
import time
import warnings
from typing import TYPE_CHECKING

import requests

from granian._loops import WorkerSignal as GranianWorkerSignal
from granian.server import Granian, Worker as GranianWorker
from gradio.exceptions import ServerFailedToStartError
from gradio.routes import App
from gradio.tunneling import Tunnel

if TYPE_CHECKING:  # Only import for type checking (to avoid circular imports).
    from gradio.blocks import Blocks

# By default, the local server will try to open on localhost, port 7860.
# If that is not available, then it will try 7861, 7862, ... 7959.
INITIAL_PORT_VALUE = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
TRY_NUM_PORTS = int(os.getenv("GRADIO_NUM_PORTS", "100"))
LOCALHOST_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
GRADIO_API_SERVER = "https://api.gradio.app/v2/tunnel-request"


class ServerWorker(GranianWorker):
    def _spawn(self, target, args):
        self.proc = threading.Thread(target=_spawn_asgi_worker, args=args, daemon=True)
        self.proc.pid = -1

    def _watch(self):
        pass

    def terminate(self):
        pass


class Server(Granian):
    def setup_signals(self):
        self._th_signal = GranianWorkerSignal()

    def _spawn_proc(self, idx, target, callback_loader, socket_loader):
        return ServerWorker(
            parent=self,
            idx=idx,
            target=target,
            args=(
                idx + 1,
                callback_loader,
                socket_loader(),
                self.loop,
                self.threads,
                self.blocking_threads,
                self.backpressure,
                self.http,
                self.http1_settings,
                self.http2_settings,
                self.websockets,
                self.loop_opt,
                self.log_enabled,
                self.log_level,
                self.log_config,
                self.log_access_format if self.log_access else None,
                self.ssl_ctx,
                {'url_path_prefix': self.url_path_prefix},
                self._th_signal
            ),
        )

    def close(self):
        self._th_signal.set()
        self.signal_handler_interrupt()


def _spawn_asgi_worker(
        worker_id,
        callback_loader,
        socket,
        loop_impl,
        threads,
        blocking_threads,
        backpressure,
        http_mode,
        http1_settings,
        http2_settings,
        websockets,
        loop_opt,
        log_enabled,
        log_level,
        log_config,
        log_access_fmt,
        ssl_ctx,
        scope_opts,
        shutdown_event,
    ):
        from granian._futures import future_watcher_wrapper
        from granian._loops import loops
        from granian._granian import ASGIWorker
        from granian.asgi import _callback_wrapper
        from granian.log import configure_logging

        configure_logging(log_level, log_config, log_enabled)

        loop = loops.get(loop_impl)
        sfd = socket.fileno()
        callback = callback_loader()

        wcallback = _callback_wrapper(callback, scope_opts, {}, log_access_fmt)
        if not loop_opt:
            wcallback = future_watcher_wrapper(wcallback)

        worker = ASGIWorker(
            worker_id,
            sfd,
            threads,
            blocking_threads,
            backpressure,
            http_mode,
            http1_settings,
            http2_settings,
            websockets,
            loop_opt,
            *ssl_ctx,
        )
        worker.serve_wth(wcallback, loop, contextvars.copy_context(), shutdown_event)


def get_first_available_port(initial: int, final: int) -> int:
    """
    Gets the first open port in a specified range of port numbers
    Parameters:
    initial: the initial value in the range of port numbers
    final: final (exclusive) value in the range of port numbers, should be greater than `initial`
    Returns:
    port: the first open port in the range
    """
    for port in range(initial, final):
        try:
            s = socket.socket()  # create a socket object
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((LOCALHOST_NAME, port))  # Bind to the port
            s.close()
            return port
        except OSError:
            pass
    raise OSError(
        f"All ports from {initial} to {final - 1} are in use. Please close a port."
    )


def configure_app(app: App, blocks: Blocks) -> App:
    auth = blocks.auth
    if auth is not None:
        if not callable(auth):
            app.auth = {account[0]: account[1] for account in auth}
        else:
            app.auth = auth
    else:
        app.auth = None
    app.blocks = blocks
    app.cwd = os.getcwd()
    app.favicon_path = blocks.favicon_path
    app.tokens = {}
    return app


def start_server(
    blocks: Blocks,
    server_name: str | None = None,
    server_port: int | None = None,
    ssl_keyfile: str | None = None,
    ssl_certfile: str | None = None,
    ssl_keyfile_password: str | None = None,
    app_kwargs: dict | None = None,
) -> tuple[str, int, str, App, Server]:
    """Launches a local server running the provided Interface
    Parameters:
        blocks: The Blocks object to run on the server
        server_name: to make app accessible on local network, set this to "0.0.0.0". Can be set by environment variable GRADIO_SERVER_NAME.
        server_port: will start gradio app on this port (if available). Can be set by environment variable GRADIO_SERVER_PORT.
        auth: If provided, username and password (or list of username-password tuples) required to access the Blocks. Can also provide function that takes username and password and returns True if valid login.
        ssl_keyfile: If a path to a file is provided, will use this as the private key file to create a local server running on https.
        ssl_certfile: If a path to a file is provided, will use this as the signed certificate for https. Needs to be provided if ssl_keyfile is provided.
        ssl_keyfile_password: If a password is provided, will use this with the ssl certificate for https.
        app_kwargs: Additional keyword arguments to pass to the gradio.routes.App constructor.

    Returns:
        port: the port number the server is running on
        path_to_local_server: the complete address that the local server can be accessed at
        app: the FastAPI app object
        server: the server object that is a subclass of uvicorn.Server (used to close the server)
    """
    if ssl_keyfile is not None and ssl_certfile is None:
        raise ValueError("ssl_certfile must be provided if ssl_keyfile is provided.")

    server_name = server_name or LOCALHOST_NAME
    url_host_name = "localhost" if server_name == "0.0.0.0" else server_name

    # Strip IPv6 brackets from the address if they exist.
    # This is needed as http://[::1]:port/ is a valid browser address,
    # but not a valid IPv6 address, so asyncio will throw an exception.
    if server_name.startswith("[") and server_name.endswith("]"):
        host = server_name[1:-1]
    else:
        host = server_name

    app = App.create_app(blocks, app_kwargs=app_kwargs)

    server_ports = (
        [server_port]
        if server_port is not None
        else range(INITIAL_PORT_VALUE, INITIAL_PORT_VALUE + TRY_NUM_PORTS)
    )

    for port in server_ports:
        try:
            # The fastest way to check if a port is available is to try to bind to it with socket.
            # If the port is not available, socket will throw an OSError.
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Really, we should be checking if (server_name, server_port) is available, but
            # socket.bind() doesn't seem to throw an OSError with ipv6 addresses, based on my testing.
            # Instead, we just check if the port is available on localhost.
            s.bind((LOCALHOST_NAME, port))
            s.close()

            # To avoid race conditions, so we also check if the port by trying to start the uvicorn server.
            # If the port is not available, this will throw a ServerFailedToStartError.
            server = Server(
                target=app,
                interface="asgi",
                address=host,
                port=port,
                log_level="warning",
                ssl_cert=ssl_certfile,
                ssl_key=ssl_keyfile,
                #ssl_key_password=ssl_keyfile_password,
            )
            server_thread = threading.Thread(
                target=server.serve,
                kwargs={'target_loader': lambda v: v},
                daemon=True
            )
            server_thread.start()
            break
        except (OSError, ServerFailedToStartError):
            pass
    else:
        raise OSError(
            f"Cannot find empty port in range: {min(server_ports)}-{max(server_ports)}. You can specify a different port by setting the GRADIO_SERVER_PORT environment variable or passing the `server_port` parameter to `launch()`."
        )

    if ssl_keyfile is not None:
        path_to_local_server = f"https://{url_host_name}:{port}/"
    else:
        path_to_local_server = f"http://{url_host_name}:{port}/"

    return server_name, port, path_to_local_server, app, server


def setup_tunnel(local_host: str, local_port: int, share_token: str) -> str:
    response = requests.get(GRADIO_API_SERVER)
    if response and response.status_code == 200:
        try:
            payload = response.json()[0]
            remote_host, remote_port = payload["host"], int(payload["port"])
            tunnel = Tunnel(
                remote_host, remote_port, local_host, local_port, share_token
            )
            address = tunnel.start_tunnel()
            return address
        except Exception as e:
            raise RuntimeError(str(e)) from e
    raise RuntimeError("Could not get share link from Gradio API Server.")


def url_ok(url: str) -> bool:
    try:
        for _ in range(5):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                r = requests.head(url, timeout=3, verify=False)
            if r.status_code in (200, 401, 302):  # 401 or 302 if auth is set
                return True
            time.sleep(0.500)
    except (ConnectionError, requests.exceptions.ConnectionError):
        return False
    return False
