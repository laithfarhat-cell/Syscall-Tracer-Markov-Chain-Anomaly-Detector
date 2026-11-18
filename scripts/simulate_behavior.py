"""Generate synthetic syscall observations only; no token is ever executed."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from core.common import atomic_text, bounded_int
from core.tracer import SyscallEvent

BOOT = ("execve", "brk", "mmap:EXEC", "close")
ANOMALY = ("mprotect:EXEC", "socket:INET", "connect", "dup2", "execve")


@dataclass(frozen=True, slots=True)
class Lab:
    training: tuple[Path, ...]
    validation: tuple[Path, ...]
    scenario: Path
    labels: Path
    first_unusual: int
    last_unusual: int


def workload(seed: int, requests: int, *, anomalous: bool = False) -> Iterator[SyscallEvent]:
    """A file-serving worker with repeatable variation and an optional labeled segment."""
    rng = random.Random(seed)
    epoch = 1_750_000_000 + seed % 1000

    def tokens() -> Iterator[str]:
        yield from BOOT
        for request in range(requests):
            yield from ("epoll_wait", "accept4", "read")
            if anomalous and request == requests // 2:
                yield from ANOMALY
            yield "openat:READ"
            yield "fstat" if rng.random() < 0.78 else "newfstatat"
            yield "write" if rng.random() < 0.88 else "sendfile"
            yield from ("close", "close")

    for i, token in enumerate(tokens()):
        yield SyscallEvent(token, pid=4200, timestamp=epoch + i * 0.01, source="simulation")


def write_lab(directory: str | Path, *, seed: int = 42, requests: int = 100) -> Lab:
    bounded_int(seed, "seed", 0, (1 << 32) - 1)
    bounded_int(requests, "requests", 1, 100_000)
    root = Path(directory)
    training = tuple(root / f"benign_{i:02d}.jsonl" for i in range(1, 5))
    validation = tuple(root / f"validation_{i:02d}.jsonl" for i in range(1, 3))
    scenario = root / "simulated_anomaly.jsonl"
    for i, path in enumerate((*training, *validation, scenario)):
        with atomic_text(path) as handle:
            for event in workload(seed + i, requests, anomalous=path == scenario):
                handle.write(event.json_line())
    first = len(BOOT) + 8 * (requests // 2) + 3 + 1
    labels = root / "scenario.json"
    with atomic_text(labels) as handle:
        json.dump(
            {
                "trace": scenario.name,
                "label": "synthetic_unusual_behavior",
                "index_base": 1,
                "first_unusual_index": first,
                "last_unusual_index": first + len(ANOMALY) - 1,
                "tokens": list(ANOMALY),
                "note": "Only normalized input records are generated; no workload or payload is executed.",
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    return Lab(training, validation, scenario, labels, first, first + len(ANOMALY) - 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/simulation"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--requests", type=int, default=100, help="worker requests per recording")
    args = parser.parse_args(argv)
    try:
        lab = write_lab(args.output_dir, seed=args.seed, requests=args.requests)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"Generated four benign, two validation and one labeled trace in {lab.labels.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
