"""Shared validation and file/stream utilities."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from io import TextIOWrapper
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, TextIO, TypeVar

DEFAULT_STREAM_LIMIT = 4096
MAX_WINDOW = 8192
StreamKey = tuple[str | None, int | None, int | None]
T = TypeVar("T")


def bounded_int(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def bounded_float(value: object, name: str, low: float, high: float) -> float:
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
    return float(value)


def stream_limit(value: int) -> int:
    return bounded_int(value, "max_streams", 1, 1_000_000)


def get_stream(states: dict[StreamKey, T], key: StreamKey, limit: int, factory: Callable[[], T]) -> T:
    try:
        return states[key]
    except KeyError:
        if len(states) >= limit:
            raise ValueError(f"trace exceeds {limit} streams; split the trace or raise max_streams") from None
        state = states[key] = factory()
        return state


@contextmanager
def atomic_binary(path: str | Path) -> Iterator[BinaryIO]:
    """Replace the destination only after writing a complete file, on the same filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".syscall-hids-", delete=False) as handle:
            tmp = Path(handle.name)
            yield handle
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


@contextmanager
def atomic_text(path: str | Path) -> Iterator[TextIO]:
    with atomic_binary(path) as raw, TextIOWrapper(raw, encoding="utf-8", newline="\n") as text:
        yield text
