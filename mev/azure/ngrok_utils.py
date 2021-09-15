from typing import Tuple

from pyngrok import ngrok


def start_tunnel(port: str) -> Tuple[str, str]:
    kill_all()

    tunnel = ngrok.connect(port, "tcp")

    _, pycharm_debug_host, pycharm_debug_port = \
        tunnel.public_url.replace("/", "").split(":")

    return pycharm_debug_host, pycharm_debug_port


def kill_all() -> None:
    ngrok.kill()