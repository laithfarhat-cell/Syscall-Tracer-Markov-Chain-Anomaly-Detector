"""Sliding likelihood detector, evidence windows and streaming JSON alert output."""

from __future__ import annotations

import json
import math
import os
import stat
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TextIO

from .common import DEFAULT_STREAM_LIMIT, MAX_WINDOW, StreamKey, bounded_float, bounded_int, get_stream, stream_limit
from .markov_model import DEFAULT_WINDOW, LikelihoodWindow, MarkovModel, Transition, window_size
from .tracer import SyscallEvent, valid_token

DEFAULT_CRITICAL = (
    "execve",
    "execveat",
    "ptrace",
    "init_module",
    "finit_module",
    "delete_module",
    "mprotect:EXEC",
    "pkey_mprotect:EXEC",
    "mmap:EXEC",
    "setuid",
    "setreuid",
    "setresuid",
    "capset",
)
DEFAULT_CRITICAL_PROBABILITY = 0.001
DEFAULT_PAIR_SPAN = 4
DRIFT = "sequence_drift"
CRITICAL = "critical_transition"
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def critical_catalog(values: Iterable[str]) -> frozenset[str]:
    if isinstance(values, str):
        raise ValueError("critical_tokens must be a list of tokens, not a string")
    tokens = tuple(values)
    if not all(valid_token(token) for token in tokens):
        raise ValueError("critical_tokens must contain valid syscall tokens")
    return frozenset(tokens)


@dataclass(frozen=True, slots=True)
class LocatedEvent:
    event: SyscallEvent
    input_index: int
    stream_index: int

    def as_dict(self) -> dict:
        return {"input_index": self.input_index, "stream_index": self.stream_index, **self.event.as_dict()}


@dataclass(frozen=True, slots=True)
class EdgeEvidence:
    transition: Transition
    destination_index: int

    def as_dict(self) -> dict:
        t = self.transition
        return {
            "context": list(t.context),
            "destination": t.destination,
            "destination_index": self.destination_index,
            "probability": t.probability,
            "observed_count": t.observed_count,
            "uses_unknown": t.uses_unknown,
            "surprise_bits": t.surprise_bits,
        }


@dataclass(frozen=True, slots=True)
class Alert:
    events: tuple[LocatedEvent, ...]
    transitions: tuple[EdgeEvidence, ...]
    score: float | None
    threshold: float
    window: int
    order: int
    triggers: tuple[str, ...]
    critical_features: tuple[str, ...]
    severity: int

    @property
    def end(self) -> LocatedEvent:
        return self.events[-1]

    def highlighted(self) -> Iterator[Transition]:
        if CRITICAL in self.triggers:
            yield self.transitions[-1].transition
        if DRIFT in self.triggers:
            yield max(self.transitions, key=lambda edge: edge.transition.surprise_bits).transition

    def as_dict(self) -> dict:
        event = self.end.event
        record = {
            "ecs": {"version": "8.11.0"},
            "event": {
                "kind": "alert",
                "category": ["intrusion_detection"],
                "type": ["indicator"],
                "dataset": "syscall_hids.markov",
                "action": "syscall-anomaly",
                "severity": self.severity,
                "risk_score_norm": self.severity,
            },
            "rule": {"id": "syscall-hids.markov", "name": "Syscall behavioral anomaly"},
            "message": "Syscall behavioral anomaly: " + ", ".join(self.triggers),
            "syscall_hids": {
                "source": event.source,
                "triggers": list(self.triggers),
                "critical_features": list(self.critical_features),
                "severity_score": self.severity,
                "order": self.order,
                "score_bits": self.score,
                "threshold_bits": self.threshold,
                "window": {
                    "expected_length": self.window,
                    "complete": self.score is not None,
                    "start_input_index": self.events[0].input_index,
                    "end_input_index": self.end.input_index,
                    "start_stream_index": self.events[0].stream_index,
                    "end_stream_index": self.end.stream_index,
                    "events": [item.as_dict() for item in self.events],
                    "transitions": [item.as_dict() for item in self.transitions],
                },
            },
        }
        if event.pid is not None:
            record["process"] = {"pid": event.pid}
        if event.stream_id is not None:
            record["syscall_hids"]["stream_id"] = event.stream_id
        if event.timestamp is not None:
            record["@timestamp"] = (_EPOCH + timedelta(seconds=event.timestamp)).isoformat().replace("+00:00", "Z")
        return record


@dataclass(frozen=True, slots=True)
class Observation:
    current: LocatedEvent
    transition: Transition | None
    score: float | None
    alert: Alert | None


class _Stream:
    def __init__(self, model: MarkovModel, window: int):
        self.likelihood = LikelihoodWindow(model, window)
        self.events: deque[LocatedEvent] = deque(maxlen=window)
        self.seen = 0
        self.last_connect: int | None = None
        self.last_dup: int | None = None


class WindowDetector:
    """Process a multiplexed capture, keeping a different window for each stream."""

    def __init__(
        self,
        model: MarkovModel,
        *,
        window: int | None = None,
        threshold: float | None = None,
        critical_tokens: Iterable[str] = DEFAULT_CRITICAL,
        critical_probability: float = DEFAULT_CRITICAL_PROBABILITY,
        pair_span: int = DEFAULT_PAIR_SPAN,
        max_streams: int = DEFAULT_STREAM_LIMIT,
    ):
        self.model = model
        self._revision = model.revision
        if not model.transition_count:
            raise ValueError("profile has no full-order transitions")
        calibration = model.calibration
        self.window = window_size(
            model.order, window if window is not None else calibration.window if calibration else DEFAULT_WINDOW
        )
        if threshold is None:
            if calibration is None or calibration.window != self.window:
                raise ValueError("a calibration for this window or an explicit --threshold is required")
            threshold = calibration.threshold
        self.threshold = bounded_float(threshold, "threshold", 0.0, 1e9)
        self.critical_tokens = critical_catalog(critical_tokens)
        self.critical_probability = bounded_float(critical_probability, "critical_probability", 0.0, 1.0)
        self.pair_span = min(bounded_int(pair_span, "critical_pair_span", 0, MAX_WINDOW), self.window - 1)
        self.limit = stream_limit(max_streams)
        self._streams: dict[StreamKey, _Stream] = {}
        self._seen = 0

    def _critical_features(self, state: _Stream, transition: Transition | None, token: str) -> tuple[str, ...]:
        features = set()
        if transition is not None:
            for candidate in (transition.context[-1], token):
                if candidate in self.critical_tokens or candidate.partition(":")[0] in self.critical_tokens:
                    features.add(candidate)
        base = token.partition(":")[0]
        if base == "connect":
            if state.last_dup is not None and state.seen - state.last_dup <= self.pair_span:
                features.add("connect+dup2/dup3")
            state.last_connect = state.seen
        elif base in {"dup2", "dup3"}:
            if state.last_connect is not None and state.seen - state.last_connect <= self.pair_span:
                features.add("connect+dup2/dup3")
            state.last_dup = state.seen
        return tuple(sorted(features))

    def _alert(self, state: _Stream, score: float | None, critical: tuple[str, ...], drift: bool) -> Alert:
        triggers = ((DRIFT,) if drift else ()) + ((CRITICAL,) if critical else ())
        events = tuple(state.events)
        tokens = tuple(item.event.token for item in events)
        start = self.model.order if score is not None else 1
        evidence = tuple(
            EdgeEvidence(
                self.model.transition(tokens[max(0, i - self.model.order) : i], tokens[i]), events[i].input_index
            )
            for i in range(start, len(events))
        )
        severity = 0
        if drift and score is not None:
            excess = (score - self.threshold) / max(1.0, self.threshold)
            severity = min(79, 40 + round(20 * excess))
        if critical:
            severity = 85 if evidence[-1].transition.observed_count == 0 else 80
            if drift:
                severity += 10
        return Alert(
            events, evidence, score, self.threshold, self.window, self.model.order, triggers, critical, severity
        )

    def observe(self, event: SyscallEvent) -> Observation:
        if self._revision != self.model.revision:
            raise ValueError("profile changed while detecting; recalibrate and start a new detector")
        state = get_stream(self._streams, event.stream, self.limit, lambda: _Stream(self.model, self.window))
        self._seen += 1
        state.seen += 1
        location = LocatedEvent(event, self._seen, state.seen)
        state.events.append(location)
        transition, score = state.likelihood.push(event.token)
        features = self._critical_features(state, transition, event.token)
        critical = bool(
            features
            and transition is not None
            and (transition.observed_count == 0 or transition.probability <= self.critical_probability)
        )
        drift = (
            score is not None
            and score > self.threshold
            and not math.isclose(score, self.threshold, rel_tol=1e-12, abs_tol=1e-12)
        )
        alert = self._alert(state, score, features if critical else (), drift) if drift or critical else None
        return Observation(location, transition, score, alert)

    def scan(self, events: Iterable[SyscallEvent]) -> Iterator[Observation]:
        for event in events:
            yield self.observe(event)


class AlertWriter:
    """Flush each alert; a .jsonl/.ndjson suffix selects line-delimited SIEM output."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.line_delimited = self.path.suffix.lower() in {".jsonl", ".ndjson"}
        self.handle: TextIO | None = None
        self.count = 0

    def __enter__(self) -> AlertWriter:
        if self.handle is not None:
            raise RuntimeError("alert file is already open")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT
        for name in ("O_NOFOLLOW", "O_NONBLOCK", "O_BINARY"):
            flags |= getattr(os, name, 0)
        fd = os.open(self.path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("alert destination must be a regular file")
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            os.ftruncate(fd, 0)
            self.handle = os.fdopen(fd, "w", encoding="utf-8", newline="\n")
        except BaseException:
            os.close(fd)
            raise
        self.count = 0
        return self

    def write(self, alert: Alert) -> None:
        if self.handle is None:
            raise RuntimeError("alert file is closed")
        body = json.dumps(alert.as_dict(), allow_nan=False, separators=(",", ":"))
        prefix = "" if self.line_delimited else ",\n" if self.count else "[\n"
        self.handle.write(prefix + body + ("\n" if self.line_delimited else ""))
        self.handle.flush()
        self.count += 1

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.handle is None:
            return
        try:
            if not self.line_delimited:
                self.handle.write("\n]\n" if self.count else "[]\n")
        finally:
            self.handle.close()
            self.handle = None
