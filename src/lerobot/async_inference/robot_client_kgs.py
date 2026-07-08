"""Backward-compatible wrapper for ``robot_client_demo``.

Prefer:

    python -m lerobot.async_inference.robot_client_demo ...
"""

from lerobot.utils.import_utils import register_third_party_plugins

from .robot_client_demo import *  # noqa: F403
from .robot_client_demo import async_client

if __name__ == "__main__":
    register_third_party_plugins()
    async_client()
