from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
from unittest.mock import patch


FORBIDDEN_HELPERS = (
    "lerobot.datasets.video_utils.encode_video_frames",
    "lerobot.datasets.video_utils.concatenate_video_files",
    "lerobot.datasets.aggregate.concatenate_video_files",
    "lerobot.datasets.dataset_tools._rebuild_trimmed_dataset",
    "lerobot.datasets.dataset_tools._keep_episodes_from_video_with_av",
    "lerobot.datasets.dataset_tools._copy_and_reindex_videos",
)


class NoReencodeViolation(RuntimeError):
    """Raised when a default operation enters a forbidden video rewrite path."""


def _raise_forbidden(name: str):
    def _inner(*args, **kwargs):
        raise NoReencodeViolation(f"Forbidden video rewrite helper was called: {name}")

    return _inner


@contextmanager
def guard_no_reencode() -> Iterator[None]:
    """Patch known encode/rebuild helpers so tests can prove they are not used."""
    patches = []
    for target in FORBIDDEN_HELPERS:
        try:
            p = patch(target, side_effect=_raise_forbidden(target))
            p.start()
            patches.append(p)
        except (AttributeError, ModuleNotFoundError):
            continue
    try:
        yield
    finally:
        for p in reversed(patches):
            p.stop()
