"""Sparse k-order conditional counts, calibrated window likelihood, and JSON profiles."""

from __future__ import annotations

import json
import math
from array import array
from collections import Counter, deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import median_abs_deviation

from .common import (
    DEFAULT_STREAM_LIMIT,
    MAX_WINDOW,
    StreamKey,
    atomic_text,
    bounded_float,
    bounded_int,
    get_stream,
    stream_limit,
)
from .tracer import SyscallEvent, valid_token

UNK = "<UNK>"
MAX_ORDER = 16
DEFAULT_WINDOW = 20
MODEL_VERSION = 1
Context = tuple[str, ...]
Rows = dict[Context, Counter[str]]


def window_size(order: int, window: int) -> int:
    return bounded_int(window, f"window (must exceed order {order})", order + 1, MAX_WINDOW)


def smoothing(value: float) -> float:
    # Keep positive probabilities representable even for large serialized counts.
    return bounded_float(value, "alpha", 1e-100, 1e100)


@dataclass(frozen=True, slots=True)
class Calibration:
    window: int
    percentile: float
    threshold: float
    samples: int
    median: float
    mad: float


@dataclass(frozen=True, slots=True)
class Transition:
    context: Context
    destination: str
    probability: float
    observed_count: int
    uses_unknown: bool

    @property
    def surprise_bits(self) -> float:
        return -math.log2(self.probability) if self.probability < 1.0 else 0.0


class MarkovModel:
    """Conditional probabilities over observed syscall tokens plus a reserved <UNK>.

    Counts for short suffixes are kept for critical-transition checks during
    warm-up, never used to replace an unseen full-order row in window scores.
    """

    def __init__(self, order: int = 1, alpha: float = 0.01):
        self._order = bounded_int(order, "order", 1, MAX_ORDER)
        self._alpha = smoothing(alpha)
        self._vocab = {UNK}
        self._rows: Rows = {}
        self._totals: dict[Context, int] = {}
        self._calibration: Calibration | None = None
        self._revision = 0

    @property
    def order(self) -> int:
        return self._order

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return tuple(sorted(self._vocab))

    @property
    def transition_count(self) -> int:
        return sum(total for context, total in self._totals.items() if len(context) == self.order)

    @property
    def calibration(self) -> Calibration | None:
        return self._calibration

    @property
    def threshold(self) -> float | None:
        return self.calibration.threshold if self.calibration else None

    @property
    def revision(self) -> int:
        return self._revision

    @staticmethod
    def _count(rows: Rows, vocabulary: set[str], history: deque[str], token: str) -> None:
        vocabulary.add(token)
        previous = tuple(history)
        for offset in range(len(previous)):
            context = previous[offset:]
            row = rows.get(context)
            if row is None:
                row = rows[context] = Counter()
            row[token] += 1
        history.append(token)

    def _merge(self, rows: Rows, vocabulary: set[str], *, replace: bool = False) -> None:
        if replace:
            self._vocab = {UNK} | vocabulary
            self._rows = rows
            self._totals = {context: row.total() for context, row in rows.items()}
        elif vocabulary:
            self._vocab.update(vocabulary)
            for context, row in rows.items():
                if context in self._rows:
                    self._rows[context].update(row)
                else:
                    self._rows[context] = row
                self._totals[context] = self._totals.get(context, 0) + row.total()
        else:
            return
        self._calibration = None
        self._revision += 1

    def _count_sequences(self, sequences: Iterable[Iterable[str]]) -> tuple[Rows, set[str]]:
        rows: Rows = {}
        vocabulary: set[str] = set()
        for sequence in sequences:
            if isinstance(sequence, str):
                raise ValueError("expected an iterable of token sequences, not individual strings")
            history: deque[str] = deque(maxlen=self.order)
            for token in sequence:
                if not valid_token(token):
                    raise ValueError("invalid training token (<UNK> is reserved)")
                self._count(rows, vocabulary, history, token)
        return rows, vocabulary

    def fit(self, sequences: Iterable[Iterable[str]]) -> MarkovModel:
        """Replace counts; do not create transitions between independent sequences."""
        self._merge(*self._count_sequences(sequences), replace=True)
        return self

    def partial_fit(self, sequences: Iterable[Iterable[str]]) -> MarkovModel:
        """Add independent sequences. Recalibrate after a nonempty update."""
        self._merge(*self._count_sequences(sequences))
        return self

    def partial_fit_events(
        self, events: Iterable[SyscallEvent], *, max_streams: int = DEFAULT_STREAM_LIMIT
    ) -> MarkovModel:
        """Add one capture; keep independent contexts per logical stream."""
        limit = stream_limit(max_streams)
        histories: dict[StreamKey, deque[str]] = {}
        rows: Rows = {}
        vocabulary: set[str] = set()
        for event in events:
            history = get_stream(histories, event.stream, limit, lambda: deque(maxlen=self.order))
            self._count(rows, vocabulary, history, event.token)
        self._merge(rows, vocabulary)
        return self

    def row_total(self, context: Context) -> int:
        return self._totals.get(context, 0)

    def _mass(self, context: Context, destination: str) -> tuple[int, float]:
        row = self._rows.get(context)
        if row is None:
            return 0, 1.0 / len(self._vocab)
        n = row.get(destination, 0)
        return n, (n + self.alpha) / (self._totals[context] + self.alpha * len(self._vocab))

    def transition(self, context: Sequence[str] | str, destination: str) -> Transition:
        original = (context,) if isinstance(context, str) else tuple(context)
        if not 1 <= len(original) <= self.order:
            raise ValueError(f"context must contain between 1 and {self.order} tokens")
        mapped = tuple(token if token in self._vocab else UNK for token in original)
        target = destination if destination in self._vocab else UNK
        n, p = self._mass(mapped, target)
        return Transition(original, destination, p, n, target == UNK or UNK in mapped)

    def probability(self, context: Sequence[str] | str, destination: str) -> float:
        return self.transition(context, destination).probability

    def observed_transitions(self) -> Iterator[tuple[Context, str, int]]:
        """Yield observed full-order edges only, without materializing smoothed edges."""
        for context, row in self._rows.items():
            if len(context) == self.order:
                for destination, count in row.items():
                    yield context, destination, count

    def contexts(self) -> tuple[Context, ...]:
        if self.order == 1:
            return tuple((token,) for token in self.vocabulary)
        return tuple(sorted(context for context in self._rows if len(context) == self.order))

    def transition_matrix(
        self, contexts: Iterable[Sequence[str]] | None = None, destinations: Iterable[str] | None = None
    ) -> np.ndarray:
        """Dense diagnostic view; unselected columns are omitted, not renormalized."""
        rows = self.contexts() if contexts is None else tuple(tuple(row) for row in contexts)
        cols = self.vocabulary if destinations is None else tuple(destinations)
        if len(set(cols)) != len(cols) or any(token not in self._vocab for token in cols):
            raise ValueError("matrix columns must be distinct vocabulary tokens")
        matrix = np.empty((len(rows), len(cols)), dtype=np.float64)
        col_index = {token: index for index, token in enumerate(cols)}
        for i, context in enumerate(rows):
            if len(context) != self.order:
                raise ValueError("matrix rows must be full-order contexts")
            row = self._rows.get(context)
            if row is None:
                matrix[i].fill(1.0 / len(self._vocab))
                continue
            denominator = self._totals[context] + self.alpha * len(self._vocab)
            matrix[i].fill(self.alpha / denominator)
            for token, count in row.items():
                if token in col_index:
                    matrix[i, col_index[token]] = (count + self.alpha) / denominator
        return matrix

    def window_score(self, tokens: Sequence[str]) -> float:
        """Direct, batch computation in bits per conditional transition."""
        window_size(self.order, len(tokens))
        return math.fsum(
            self.transition(tokens[i - self.order : i], tokens[i]).surprise_bits for i in range(self.order, len(tokens))
        ) / (len(tokens) - self.order)

    def calibrate(
        self,
        traces: Iterable[Iterable[SyscallEvent]],
        *,
        window: int = DEFAULT_WINDOW,
        percentile: float = 99.0,
        max_streams: int = DEFAULT_STREAM_LIMIT,
    ) -> Calibration:
        """Calibrate on complete windows of independent, held-out benign captures."""
        window_size(self.order, window)
        percentile = bounded_float(percentile, "percentile", 0.0, 100.0)
        limit = stream_limit(max_streams)
        if not self.transition_count:
            raise ValueError(f"training did not contain any order-{self.order} transitions")
        scores = array("d")
        for events in traces:
            states: dict[StreamKey, LikelihoodWindow] = {}
            for event in events:
                state = get_stream(states, event.stream, limit, lambda: LikelihoodWindow(self, window))
                _, score = state.push(event.token)
                if score is not None:
                    scores.append(score)
        if not scores:
            raise ValueError(f"no complete validation windows (need {window} calls in one validation stream)")
        values = np.frombuffer(scores, dtype=np.float64)
        self._calibration = Calibration(
            window,
            percentile,
            float(np.percentile(values, percentile, method="linear")),
            len(scores),
            float(np.median(values)),
            float(median_abs_deviation(values, scale=1.0)),
        )
        return self._calibration

    def to_dict(self) -> dict:
        return {
            "version": MODEL_VERSION,
            "order": self.order,
            "alpha": self.alpha,
            "vocabulary": list(self.vocabulary),
            "rows": [{"context": list(context), "next": dict(row)} for context, row in sorted(self._rows.items())],
            "calibration": asdict(self.calibration) if self.calibration else None,
        }

    @classmethod
    def from_dict(cls, payload: object) -> MarkovModel:
        if (
            not isinstance(payload, dict)
            or type(payload.get("version")) is not int
            or payload["version"] != MODEL_VERSION
        ):
            raise ValueError(f"invalid or unsupported profile version (expected {MODEL_VERSION})")
        model = cls(payload.get("order"), payload.get("alpha"))
        vocab = payload.get("vocabulary")
        if not isinstance(vocab, list) or not all(token == UNK or valid_token(token) for token in vocab):
            raise ValueError("invalid profile vocabulary")
        model._vocab = set(vocab)
        if len(model._vocab) != len(vocab) or UNK not in model._vocab:
            raise ValueError("profile vocabulary must be unique and include <UNK>")
        known = model._vocab - {UNK}
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError("invalid profile rows")
        for record in rows:
            if not isinstance(record, dict):
                raise ValueError("invalid profile row")
            context, counts = record.get("context"), record.get("next")
            if (
                not isinstance(context, list)
                or not 1 <= len(context) <= model.order
                or any(not valid_token(token) or token not in known for token in context)
            ):
                raise ValueError("invalid profile context")
            ctx = tuple(context)
            if ctx in model._rows or not isinstance(counts, dict) or not counts:
                raise ValueError("duplicate, empty or invalid profile row")
            row: Counter[str] = Counter()
            for token, count in counts.items():
                if token not in known:
                    raise ValueError("profile destination is not in the observed vocabulary")
                row[token] = bounded_int(count, "transition count", 1, (1 << 63) - 1)
            model._rows[ctx] = row
            model._totals[ctx] = row.total()
        if (data := payload.get("calibration")) is not None:
            if not isinstance(data, dict) or not model.transition_count:
                raise ValueError("invalid calibration or profile has no full-order transitions")
            model._calibration = Calibration(
                window_size(model.order, data.get("window")),
                bounded_float(data.get("percentile"), "percentile", 0.0, 100.0),
                bounded_float(data.get("threshold"), "threshold", 0.0, 1e9),
                bounded_int(data.get("samples"), "samples", 1, (1 << 63) - 1),
                bounded_float(data.get("median"), "median", 0.0, 1e9),
                bounded_float(data.get("mad"), "mad", 0.0, 1e9),
            )
        return model

    def save(self, path: str | Path) -> None:
        with atomic_text(path) as handle:
            json.dump(self.to_dict(), handle, sort_keys=True, allow_nan=False, separators=(",", ":"))
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> MarkovModel:
        try:
            with Path(path).open(encoding="utf-8") as handle:
                return cls.from_dict(json.load(handle))
        except (ValueError, RecursionError) as exc:
            raise ValueError(f"{path}: {exc}") from exc


class LikelihoodWindow:
    """Shared rolling computation for calibration and detection, per stream."""

    def __init__(self, model: MarkovModel, window: int):
        self.model = model
        self.revision = model.revision
        self.history: deque[str] = deque(maxlen=model.order)
        self.capacity = window_size(model.order, window) - model.order
        self.costs: deque[float] = deque()
        self.total = 0.0
        self.steps = 0

    def push(self, token: str) -> tuple[Transition | None, float | None]:
        if self.model.revision != self.revision:
            raise ValueError("profile changed while scoring; recalibrate and start a new detector")
        transition = self.model.transition(self.history, token) if self.history else None
        score = None
        if len(self.history) == self.model.order and transition is not None:
            if len(self.costs) == self.capacity:
                self.total -= self.costs.popleft()
            cost = transition.surprise_bits
            self.costs.append(cost)
            self.total += cost
            self.steps += 1
            if self.steps == self.capacity:
                # Bound accumulated floating-point error without rescoring each window.
                self.total = math.fsum(self.costs)
                self.steps = 0
            if len(self.costs) == self.capacity:
                score = max(0.0, self.total / self.capacity)
        self.history.append(token)
        return transition, score
