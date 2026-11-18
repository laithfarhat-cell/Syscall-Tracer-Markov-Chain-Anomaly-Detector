"""Headless probability heatmap, score timeline, and Graphviz export."""

from __future__ import annotations

import heapq
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from core.common import StreamKey, atomic_binary, atomic_text, bounded_int
from core.detector import CRITICAL, DRIFT, Observation
from core.markov_model import UNK, Context, MarkovModel

HEATMAP = "markov_heatmap.png"
TIMELINE = "anomaly_timeline.png"
GRAPH = "markov_graph.dot"
PLOT_FILES = (HEATMAP, TIMELINE, GRAPH)
BLUE = "#2470ae"
RED = "#bc2c44"
INK = "#15283e"
SLATE = "#466077"
PAPER = "#f7fafc"
GOLD = "#a06616"
EdgeKey = tuple[Context, str]


@dataclass(frozen=True, slots=True)
class PlotPoint:
    index: int
    timestamp: float | None
    score: float | None


@dataclass(frozen=True, slots=True)
class PlotAlert:
    point: PlotPoint
    triggers: tuple[str, ...]
    edges: tuple[EdgeKey, ...]


class Diagnostics:
    """Bounded plot recording. JSON alerts, not this recorder, preserve all evidence."""

    def __init__(self, sample_limit: int = 8000, alert_limit: int = 2000):
        self.sample_limit = bounded_int(sample_limit, "sample_limit", 2, 1_000_000)
        self.alert_limit = bounded_int(alert_limit, "alert_limit", 1, 1_000_000)
        self.points: list[PlotPoint] = []
        self._markers: list[tuple[tuple[int, float, int], PlotAlert]] = []
        self.stride = 1
        self.scored = 0
        self.alert_count = 0
        self.clock_ok = True
        self.start_time: float | None = None
        self._last_time: float | None = None
        self._first_stream: StreamKey | None = None
        self.multi_stream = False

    def _keep_peak(self, bucketed: list[PlotPoint], point: PlotPoint) -> None:
        if bucketed and (bucketed[-1].index - 1) // self.stride == (point.index - 1) // self.stride:
            if point.score > bucketed[-1].score:
                bucketed[-1] = point
        else:
            bucketed.append(point)

    def record(self, observation: Observation) -> None:
        current = observation.current
        ts = current.event.timestamp
        if self._first_stream is None:
            self._first_stream = current.event.stream
            self.start_time = ts
        elif current.event.stream != self._first_stream:
            self.multi_stream = True
        if ts is None or (self._last_time is not None and ts < self._last_time):
            self.clock_ok = False
        self._last_time = ts
        point = PlotPoint(current.input_index, ts, observation.score)
        if point.score is not None:
            self.scored += 1
            self._keep_peak(self.points, point)
            while len(self.points) > self.sample_limit:
                self.stride *= 2
                merged: list[PlotPoint] = []
                for retained in self.points:
                    self._keep_peak(merged, retained)
                self.points = merged
        if alert := observation.alert:
            self.alert_count += 1
            marker = PlotAlert(
                point, alert.triggers, tuple(dict.fromkeys((t.context, t.destination) for t in alert.highlighted()))
            )
            rank = (alert.severity, point.score if point.score is not None else -1.0, point.index)
            if len(self._markers) < self.alert_limit:
                heapq.heappush(self._markers, (rank, marker))
            else:
                heapq.heappushpop(self._markers, (rank, marker))

    @property
    def markers(self) -> tuple[PlotAlert, ...]:
        return tuple(sorted((marker for _, marker in self._markers), key=lambda marker: marker.point.index))

    def save(self, model: MarkovModel, threshold: float, window: int, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        plot_heatmap(model, root / HEATMAP)
        plot_timeline(self, threshold, window, model.order, root / TIMELINE)
        export_dot(model, root / GRAPH, (edge for marker in self.markers for edge in marker.edges))


def _figure(title: str, description: str, *, size: tuple[float, float]) -> tuple[Figure, Axes]:
    figure = Figure(figsize=size, facecolor=PAPER, layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(facecolor="white")
    ax.set_title(f"{title}\n{description}", loc="left", color=INK, fontsize=13, pad=15, linespacing=1.6)
    ax.tick_params(colors=SLATE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#c7d6e4")
    return figure, ax


def _label(context: Context) -> str:
    return ("… → " if len(context) > 4 else "") + " → ".join(context[-4:])


def plot_heatmap(model: MarkovModel, path: str | Path, *, limit: int = 18) -> None:
    """Plot up to `limit` contexts and tokens using their original probabilities."""
    limit = bounded_int(limit, "heatmap limit", 2, 40)
    frequency: Counter[str] = Counter()
    for _, destination, count in model.observed_transitions():
        frequency[destination] += count
    vocab = model.vocabulary
    if len(vocab) <= limit:
        cols = tuple(t for t in vocab if t != UNK) + (UNK,)
    else:
        cols = tuple(heapq.nsmallest(limit - 1, (t for t in vocab if t != UNK), key=lambda t: (-frequency[t], t))) + (
            UNK,
        )
    model_contexts = model.contexts()
    if model.order == 1:
        rows = tuple((token,) for token in cols)
    else:
        rows = heapq.nsmallest(limit, model_contexts, key=lambda c: (-model.row_total(c), c))
    if not rows:
        raise ValueError("there are no full-order contexts to plot")
    matrix = model.transition_matrix(rows, cols)
    low, high = float(matrix.min()), float(matrix.max())
    norm = LogNorm(vmin=low, vmax=high) if low < high else Normalize(vmin=0, vmax=1)
    probability_scale = "log color scale" if low < high else "linear color scale"
    selection = " · top rows/columns only" if len(rows) < len(model_contexts) or len(cols) < len(vocab) else ""
    annotate = max(matrix.shape) <= 13
    subtitle = f"Order {model.order} · add-α = {model.alpha:g} · {probability_scale}{selection}"
    if annotate:
        subtitle += " · labels on observed edges"
    fig, ax = _figure(
        "Syscall transition probabilities",
        subtitle,
        size=(max(7.6, 0.52 * len(cols)), max(5.4, 0.39 * len(rows) + 2.2)),
    )
    image = ax.imshow(matrix, cmap="cividis", norm=norm, aspect="auto", interpolation="nearest")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.88, pad=0.035)
    colorbar.set_label("Conditional probability", color=INK)
    ax.set_xticks(np.arange(len(cols)), cols, rotation=46, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(rows)), [_label(row) for row in rows])
    ax.set_xlabel("Destination syscall token", color=INK)
    ax.set_ylabel("Current syscall" if model.order == 1 else "Preceding context", color=INK)
    if annotate:
        for i, row in enumerate(matrix):
            for j, p in enumerate(row):
                if model.transition(rows[i], cols[j]).observed_count:
                    label = "≈1" if 0.995 <= p < 1.0 else f"{p:.2g}"
                    ax.text(
                        j,
                        i,
                        label,
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="#142238" if norm(p) > 0.57 else "#f8fbff",
                    )
    _save_png(fig, path)


def _save_png(figure: Figure, path: str | Path) -> None:
    with atomic_binary(path) as handle:
        figure.savefig(handle, dpi=180, format="png")
    figure.clear()


def plot_timeline(diagnostics: Diagnostics, threshold: float, window: int, order: int, path: str | Path) -> None:
    markers = diagnostics.markers
    description = (
        f"{window} calls / window · order {order} · {diagnostics.scored:,} scored · {diagnostics.alert_count:,} alerted"
    )
    if diagnostics.stride > 1:
        description += f" · max per {diagnostics.stride:,} input calls"
    if len(markers) < diagnostics.alert_count:
        description += f" · {len(markers):,} markers retained"
    fig, ax = _figure("Behavioral anomaly timeline", description, size=(11.0, 5.1))
    clock = diagnostics.clock_ok and diagnostics.start_time is not None

    def x(point: PlotPoint) -> float:
        if clock and point.timestamp is not None:
            return point.timestamp - diagnostics.start_time
        return float(point.index)

    if diagnostics.points:
        xs = [x(point) for point in diagnostics.points]
        ys = [point.score for point in diagnostics.points]
        if diagnostics.multi_stream or len(xs) == 1:
            ax.scatter(xs, ys, s=15, c=BLUE, alpha=0.68, label="Window surprise")
        else:
            ax.plot(xs, ys, color=BLUE, linewidth=1.65, label="Window surprise")
            ax.fill_between(xs, ys, 0, color=BLUE, alpha=0.08)
    else:
        ax.text(0.5, 0.49, "No complete windows in this capture", transform=ax.transAxes, ha="center", color=SLATE)
    ax.axhline(threshold, color=GOLD, linestyle=(0, (5, 3)), linewidth=1.6, label=f"Threshold · {threshold:.3g} bits")
    for critical in (False, True):
        selected = [m.point for m in markers if (CRITICAL in m.triggers if critical else m.triggers == (DRIFT,))]
        complete = [p for p in selected if p.score is not None]
        warmup = [p for p in selected if p.score is None]
        if complete:
            ax.scatter(
                [x(p) for p in complete],
                [p.score for p in complete],
                s=42 if critical else 19,
                c=RED,
                marker="^" if critical else "o",
                edgecolors="white",
                linewidths=0.4,
                zorder=5,
                label="Critical transition" if critical else "Sequence drift",
            )
        if warmup:
            ax.scatter(
                [x(p) for p in warmup],
                [0.05] * len(warmup),
                color=RED,
                marker="|",
                s=105,
                transform=ax.get_xaxis_transform(),
                zorder=6,
                label="Critical (window warming up)",
            )
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Elapsed trace time (s)" if clock else "Input syscall index", color=INK)
    if not clock:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Surprise (bits per conditional transition)", color=INK)
    ax.grid(axis="y", color="#dce6ee", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="white", edgecolor="#dce6ee")
    _save_png(fig, path)


def export_dot(model: MarkovModel, path: str | Path, highlighted: Iterable[EdgeKey] = (), *, limit: int = 120) -> None:
    """Blue observed baseline edges; red evidence, including previously unseen raw tokens."""
    limit = bounded_int(limit, "graph edge limit", 1, 1000)
    training = heapq.nsmallest(limit, model.observed_transitions(), key=lambda row: (-row[2], row[0], row[1]))
    ranked = heapq.nsmallest(limit, Counter(highlighted).items(), key=lambda row: (-row[1], row[0]))
    suspicious = {key for key, _ in ranked}
    edges = sorted({(context, destination) for context, destination, _ in training} | suspicious)
    nodes = sorted({node for context, dest in edges for node in (context, (*context[1:], dest))})
    ids = {node: f"s{index}" for index, node in enumerate(nodes)}
    with atomic_text(path) as dot:
        dot.write(
            'digraph SyscallHIDS {\n  rankdir=LR;\n  graph [bgcolor="#f7fafc", labelloc=t, '
            f'label="Order {model.order} · Blue: observed baseline | Red: alert evidence"'
            '];\n  node [shape=box, style="rounded,filled", fillcolor="white", color="#a6bccd", '
            'fontname="DejaVu Sans"];\n  edge [fontname="DejaVu Sans", fontsize=9];\n'
        )
        for node in nodes:
            dot.write(f"  {ids[node]} [label={json.dumps(' → '.join(node), ensure_ascii=False)}];\n")
        for context, dest in edges:
            color = RED if (context, dest) in suspicious else BLUE
            width = 2.4 if color == RED else 1.25
            p = model.probability(context, dest)
            dot.write(
                f"  {ids[context]} -> {ids[(*context[1:], dest)]} "
                f'[label="{p:.3g}", color="{color}", fontcolor="{color}", penwidth={width}];\n'
            )
        dot.write("}\n")
