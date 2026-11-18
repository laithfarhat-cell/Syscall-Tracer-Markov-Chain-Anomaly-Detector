"""Rich telemetry with bounded refresh rate; separate from alert delivery."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .detector import CRITICAL, DRIFT, Alert, Observation, WindowDetector


@dataclass(slots=True)
class Counters:
    events: int = 0
    windows: int = 0
    alerts: int = 0
    drift: int = 0
    critical: int = 0
    segment_alerts: int = 0

    def record(self, observation: Observation, segment: tuple[int, int] | None) -> None:
        self.events += 1
        self.windows += observation.score is not None
        if alert := observation.alert:
            self.alerts += 1
            self.drift += DRIFT in alert.triggers
            self.critical += CRITICAL in alert.triggers
            if segment is not None and segment[0] <= alert.end.input_index <= segment[1]:
                self.segment_alerts += 1


class Dashboard:
    def __init__(self, live: Live, detector: WindowDetector):
        self.live = live
        self.detector = detector
        self.recent: deque[Alert] = deque(maxlen=3)
        self.next_refresh = 0.0

    def update(self, observation: Observation, counts: Counters) -> None:
        if observation.alert:
            self.recent.append(observation.alert)
        now = time.monotonic()
        if now < self.next_refresh:
            return
        self.next_refresh = now + 0.15
        event = observation.current.event
        stats = Table.grid(padding=(0, 2))
        stats.add_column(style="bold cyan")
        stats.add_column()
        stats.add_row(
            "Stream",
            f"{event.source or 'local'} / {event.pid or 'anonymous'} · event {observation.current.stream_index:,}",
        )
        transition = observation.transition
        if transition:
            step = " → ".join((*transition.context, transition.destination))
            stats.add_row("Transition", f"{step}  (p={transition.probability:.4g})")
        else:
            stats.add_row("Current token", event.token)
        score = observation.score
        tau = self.detector.threshold
        if score is None:
            stats.add_row("Surprise", f"Warming up · {self.detector.window} calls required")
        else:
            cells = 24
            filled = min(cells, round(cells * score / max(1.5 * tau, 0.01)))
            bar = Text("■" * filled, style="bold red" if score > tau else "green")
            bar.append("·" * (cells - filled), style="dim")
            bar.append(f"  {score:.3f} bits  /  τ {tau:.3f}")
            stats.add_row("Surprise", bar)
        stats.add_row("Totals", f"{counts.events:,} calls · {counts.windows:,} windows · {counts.alerts:,} alerts")
        panels = [Panel(stats, title="SYSCALL HIDS  /  TELEMETRY", border_style="cyan", expand=True)]
        if self.recent:
            alerts = Text()
            for item in reversed(self.recent):
                edge = item.transitions[-1].transition
                alerts.append(f"{item.severity:>2}  #{item.end.input_index:,}  ", style="bold red")
                alerts.append(f"{', '.join(item.triggers)} · {edge.context[-1]} → {edge.destination}\n")
            panels.append(Panel(alerts, title="RECENT ALERTS", border_style="red", expand=True))
        self.live.update(Group(*panels), refresh=True)
