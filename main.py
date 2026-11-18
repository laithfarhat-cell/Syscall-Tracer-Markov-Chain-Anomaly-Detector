"""Command-line entry point for the syscall Markov HIDS."""

from __future__ import annotations

import argparse
import glob
import signal
import sys
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager, nullcontext
from itertools import chain, islice
from pathlib import Path
from threading import current_thread, main_thread

from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table

from core.common import atomic_text
from core.config import Settings, load_settings
from core.dashboard import Counters, Dashboard
from core.detector import AlertWriter, WindowDetector
from core.markov_model import DEFAULT_WINDOW, MarkovModel, window_size
from core.tracer import SyscallEvent, TraceError, live_strace, read_trace
from scripts.simulate_behavior import write_lab

FORMATS = ("auto", "jsonl", "strace", "auditd")
DEFAULT_ALERTS = Path("syscall_alerts.json")


class _Termination(KeyboardInterrupt):
    """Route SIGTERM through the same cleanup path as an interactive interrupt."""


def _signal_status(exc: KeyboardInterrupt) -> int:
    return 128 + (signal.SIGTERM if isinstance(exc, _Termination) else signal.SIGINT)


@contextmanager
def _shutdown_signals() -> Iterator[None]:
    if sys.platform == "win32" or current_thread() is not main_thread():
        yield
        return

    def stop(signum, frame):
        raise _Termination

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _add_capture_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pid", type=int, help="attach to an existing Linux process")
    parser.add_argument("--syscalls", help="strace filter (all, or comma-separated syscall names)")
    parser.add_argument("--no-follow", action="store_true", help="do not follow threads and children")


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--order", type=int, help="Markov order (default: 1)")
    parser.add_argument("--alpha", type=float, help="positive additive smoothing (default: 0.01)")
    parser.add_argument("--window", type=int, help="calibration window size (default: 20)")
    parser.add_argument("--percentile", type=float, help="held-out percentile (default: 99)")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="syscall-hids",
        description="Trace Linux syscalls, profile benign workloads, and detect behavioral drift.",
        epilog="Place an optional --config before the subcommand, and a spawned program after --.",
    )
    parser.add_argument("--config", type=Path, help="optional YAML settings")
    commands = parser.add_subparsers(dest="command", required=True)

    trace = commands.add_parser("trace", help="capture normalized JSONL from Linux strace")
    trace.add_argument("-o", "--output", type=Path, required=True, help="normalized JSONL destination")
    _add_capture_args(trace)

    train = commands.add_parser("train", help="fit and calibrate a Markov profile from benign recordings")
    train.add_argument("-i", "--input", nargs="+", required=True, help="training files or globs")
    train.add_argument("--validation", nargs="+", help="independent benign files or globs for calibration")
    train.add_argument("--format", choices=FORMATS, default="auto", help="format for training and validation files")
    train.add_argument("--model", type=Path, required=True, help="output JSON profile")
    train.add_argument("--update", type=Path, help="add new counts to an existing JSON profile, then recalibrate")
    _add_model_args(train)

    monitor = commands.add_parser("monitor", help="score a recording or a live Linux process")
    monitor.add_argument("--model", type=Path, required=True)
    monitor.add_argument("--input", required=True, metavar="live|PATH", help="capture live or score a file")
    monitor.add_argument("--format", choices=FORMATS, default="auto", help="recorded input format")
    monitor.add_argument("--window", type=int, help="window size (default: calibration window)")
    monitor.add_argument("--threshold", type=float, help="override calibration with a surprise threshold in bits")
    monitor.add_argument("--alerts", type=Path, default=DEFAULT_ALERTS, help="alerts (.json array or .jsonl stream)")
    monitor.add_argument("--plot", action="store_true", help="write a heatmap, timeline and Graphviz file on exit")
    monitor.add_argument("--plot-dir", type=Path, help="chart directory (default: next to alert output)")
    monitor.add_argument("--quiet", action="store_true", help="disable interactive dashboard (keep summary)")
    _add_capture_args(monitor)

    simulate = commands.add_parser("simulate", help="run the detector end to end on inert synthetic records")
    simulate.add_argument("--output-dir", type=Path, default=Path("artifacts/simulation"))
    simulate.add_argument("--seed", type=int, default=42)
    simulate.add_argument("--requests", type=int, default=100, help="worker requests per recording")
    simulate.add_argument("--quiet", action="store_true", help="disable interactive dashboard (keep summary)")
    _add_model_args(simulate)
    return parser


def _line(console: Console, message: str, *, style: str | None = None) -> None:
    # Non-UTF-8 pipes on Windows and legacy terminals must not turn successful
    # analysis into a reporting error. Artifact files always use UTF-8.
    encoding = console.encoding
    safe = message.encode(encoding, errors="backslashreplace").decode(encoding)
    console.print(safe, style=style, markup=False, highlight=False)


def _coalesce(*values):
    return next((value for value in values if value is not None), None)


def _identity(path: Path) -> set[tuple]:
    keys: set[tuple] = {("path", path.resolve())}
    try:
        info = path.stat()
        if info.st_ino:
            keys.add(("inode", info.st_dev, info.st_ino))
    except FileNotFoundError:
        pass
    return keys


def _no_overlaps(paths: Sequence[Path], *, purpose: str) -> None:
    used: set[tuple] = set()
    for path in paths:
        keys = _identity(path)
        if keys & used:
            raise ValueError(f"{purpose}: overlapping file {path}")
        used.update(keys)


def _file_list(patterns: Sequence[str]) -> list[Path]:
    paths = []
    for pattern in patterns:
        if Path(pattern).exists():
            matches = [pattern]
        else:
            matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            raise ValueError(f"no trace files match {pattern}")
        for match in matches:
            path = Path(match)
            if not path.is_file():
                raise ValueError(f"not a trace file: {path}")
            paths.append(path)
    _no_overlaps(paths, purpose="trace inputs")
    return paths


def _required(events: Iterator[SyscallEvent], description: str) -> Iterator[SyscallEvent]:
    with closing(events):
        first = next(events, None)
        if first is None:
            raise TraceError(f"no syscalls found in {description}; check the format, filter and capture policy")
        yield first
        yield from events


def _file_events(path: Path, format: str, settings: Settings) -> Iterator[SyscallEvent]:
    return _required(read_trace(path, format, max_streams=settings.max_streams), str(path))


def _live(args: argparse.Namespace, target: Sequence[str], settings: Settings) -> Iterator[SyscallEvent]:
    return _required(
        live_strace(
            command=target,
            pid=args.pid,
            syscalls=args.syscalls or "all",
            follow=not args.no_follow,
            max_streams=settings.max_streams,
        ),
        "live capture",
    )


def _trace(args: argparse.Namespace, target: Sequence[str], settings: Settings, console: Console) -> int:
    if args.config is not None:
        _no_overlaps([args.config, args.output], purpose="trace output")
    count = 0
    status = 0
    with atomic_text(args.output) as handle:
        try:
            with closing(_live(args, target, settings)) as capture:
                for event in capture:
                    handle.write(event.json_line())
                    count += 1
        except KeyboardInterrupt as exc:
            if not count:
                raise
            status = _signal_status(exc)
    _line(console, f"Captured {count:,} normalized syscalls -> {args.output}")
    return status


def _model_settings(
    args: argparse.Namespace, settings: Settings, base: MarkovModel | None = None
) -> tuple[MarkovModel, int, float]:
    order = _coalesce(args.order, settings.order)
    alpha = _coalesce(args.alpha, settings.alpha)
    if base is None:
        base = MarkovModel(_coalesce(order, 1), _coalesce(alpha, 0.01))
    elif (order is not None and order != base.order) or (alpha is not None and alpha != base.alpha):
        raise ValueError("an incremental update cannot change the profile's order or alpha")
    calibration = base.calibration
    window = window_size(
        base.order, _coalesce(args.window, settings.window, calibration.window if calibration else DEFAULT_WINDOW)
    )
    percentile = _coalesce(args.percentile, settings.percentile, calibration.percentile if calibration else 99.0)
    return base, window, percentile


def _report_profile(console: Console, model: MarkovModel, path: Path) -> None:
    cal = model.calibration
    if cal is None:
        raise RuntimeError("expected a calibrated model")
    _line(
        console,
        f"Order {model.order} | {len(model.vocabulary):,} tokens including <UNK> | "
        f"{model.transition_count:,} full-order transitions -> {path}",
    )
    _line(
        console,
        f"Threshold: {cal.threshold:.6g} bits | p{cal.percentile:g} | W={cal.window} | "
        f"{cal.samples:,} validation windows (median {cal.median:.4g}, MAD {cal.mad:.4g})",
    )
    if cal.samples < 100:
        console.print(
            "Few validation windows. Verify the threshold on more independent benign workload captures.", style="yellow"
        )


def _train(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    training = _file_list(args.input)
    validation = _file_list(args.validation) if args.validation else []
    _no_overlaps([*training, *validation, args.model, *([args.config] if args.config else [])], purpose="train files")
    if args.update is not None:
        _no_overlaps([*training, *validation, args.update], purpose="update inputs")
    model, window, percentile = _model_settings(args, settings, MarkovModel.load(args.update) if args.update else None)
    opts = dict(window=window, percentile=percentile, max_streams=settings.max_streams)
    if not validation and len(training) == 1:
        path = training[0]
        total = sum(1 for _ in _file_events(path, args.format, settings))
        split = total * 4 // 5
        console.print(f"Holding out the final {total - split:,} calls of the recording for calibration.")
        with closing(_file_events(path, args.format, settings)) as capture:
            model.partial_fit_events(islice(capture, split), max_streams=settings.max_streams)
            model.calibrate((capture,), **opts)
    else:
        if not validation:
            holdout = (len(training) + 4) // 5
            training, validation = training[:-holdout], training[-holdout:]
            console.print(f"Holding out the last {len(validation)} input file(s) for calibration.")
        _fit_files(model, training, validation, args.format, settings, window, percentile)
    model.save(args.model)
    _report_profile(console, model, args.model)
    return 0


def _fit_files(
    model: MarkovModel,
    training: Sequence[Path],
    validation: Sequence[Path],
    format: str,
    settings: Settings,
    window: int,
    percentile: float,
) -> None:
    for path in training:
        with closing(_file_events(path, format, settings)) as capture:
            model.partial_fit_events(capture, max_streams=settings.max_streams)

    def traces() -> Iterator[Iterator[SyscallEvent]]:
        for path in validation:
            with closing(_file_events(path, format, settings)) as capture:
                yield capture

    with closing(traces()) as captures:
        model.calibrate(captures, window=window, percentile=percentile, max_streams=settings.max_streams)


def _detector(
    model: MarkovModel, settings: Settings, *, window: int | None = None, threshold: float | None = None
) -> WindowDetector:
    return WindowDetector(
        model,
        window=_coalesce(window, settings.window),
        threshold=threshold,
        max_streams=settings.max_streams,
        critical_tokens=settings.critical_tokens,
        critical_probability=settings.critical_probability,
        pair_span=settings.critical_pair_span,
    )


def _monitor_run(
    events: Iterator[SyscallEvent],
    detector: WindowDetector,
    alert_path: Path,
    console: Console,
    *,
    quiet: bool,
    plot_dir: Path | None,
    segment: tuple[int, int] | None = None,
) -> int:
    diagnostics = None
    if plot_dir is not None:
        from visualizers.graph_plotter import Diagnostics

        diagnostics = Diagnostics()
    counts = Counters()
    interactive = console.is_terminal and not quiet
    status = 0
    try:
        with closing(events):
            first = next(events)  # required() validates nonempty before opening the alert destination
            live = Live(console=console, auto_refresh=False, transient=True) if interactive else None
            with AlertWriter(alert_path) as writer, live if live else nullcontext():
                dashboard = Dashboard(live, detector) if live else None
                for sample in detector.scan(chain((first,), events)):
                    if sample.alert is not None:
                        writer.write(sample.alert)
                    counts.record(sample, segment)
                    if diagnostics is not None:
                        diagnostics.record(sample)
                    if dashboard is not None:
                        dashboard.update(sample, counts)
    except KeyboardInterrupt as exc:
        if not counts.events:
            raise
        status = _signal_status(exc)
    if diagnostics is not None:
        diagnostics.save(detector.model, detector.threshold, detector.window, plot_dir)
    table = Table(
        title="Behavioral monitoring summary",
        header_style="bold cyan",
        box=box.SIMPLE_HEAVY if console.is_terminal else box.ASCII,
    )
    for label in ("Calls", "Scored", "Alerts", "Drift", "Critical"):
        table.add_column(label, justify="right")
    row = [counts.events, counts.windows, counts.alerts, counts.drift, counts.critical]
    if segment is not None:
        table.add_column("Alert ends in labeled segment", justify="right")
        row.append(counts.segment_alerts)
    table.add_row(*(f"{number:,}" for number in row))
    console.print(table)
    _line(console, f"Alerts -> {alert_path}")
    if plot_dir is not None:
        _line(console, f"Heatmap, timeline and state graph -> {plot_dir}")
    return status


def _monitor(args: argparse.Namespace, target: Sequence[str], settings: Settings, console: Console) -> int:
    inputs = [args.model, *([args.config] if args.config else [])]
    if args.input == "live":
        events = _live(args, target, settings)
    else:
        source = Path(args.input)
        inputs.append(source)
        events = _file_events(source, args.format, settings)
    outputs = [args.alerts]
    plot_dir = (args.plot_dir or args.alerts.parent) if args.plot else None
    if plot_dir is not None:
        from visualizers.graph_plotter import PLOT_FILES

        outputs.extend(plot_dir / name for name in PLOT_FILES)
    _no_overlaps([*inputs, *outputs], purpose="monitor files")
    detector = _detector(MarkovModel.load(args.model), settings, window=args.window, threshold=args.threshold)
    return _monitor_run(events, detector, args.alerts, console, quiet=args.quiet, plot_dir=plot_dir)


def _simulate(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    model, window, percentile = _model_settings(args, settings)
    if 4 + 8 * args.requests < window:
        raise ValueError("not enough synthetic requests for the selected window; raise --requests or lower --window")
    root = args.output_dir
    lab = write_lab(root, seed=args.seed, requests=args.requests)
    path = root / "profile.json"
    _fit_files(model, lab.training, lab.validation, "jsonl", settings, window, percentile)
    model.save(path)
    _line(console, f"Inert synthetic segment: calls {lab.first_unusual}-{lab.last_unusual} (see {lab.labels}).")
    _report_profile(console, model, path)
    return _monitor_run(
        _file_events(lab.scenario, "jsonl", settings),
        _detector(model, settings, window=window),
        root / DEFAULT_ALERTS,
        console,
        quiet=args.quiet,
        plot_dir=root,
        segment=(lab.first_unusual, lab.last_unusual),
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    target: list[str] = []
    delimiter = "--" in arguments
    if delimiter:
        index = arguments.index("--")
        arguments, target = arguments[:index], arguments[index + 1 :]
    parser = make_parser()
    args = parser.parse_args(arguments)
    live = args.command == "trace" or (args.command == "monitor" and args.input == "live")
    if live:
        if bool(target) == (args.pid is not None) or (delimiter and not target):
            parser.error("give exactly one of --pid PID or -- PROGRAM [ARGS ...]")
        if args.command == "monitor" and args.format != "auto":
            parser.error("--format applies only to recorded input")
    elif delimiter or (
        args.command == "monitor" and (args.pid is not None or args.syscalls is not None or args.no_follow)
    ):
        parser.error("capture options require trace or monitor --input live")
    if args.command == "monitor" and args.plot_dir and not args.plot:
        parser.error("--plot-dir requires --plot")
    console = Console()
    try:
        with _shutdown_signals():
            settings = load_settings(args.config)
            if args.command == "trace":
                return _trace(args, target, settings, console)
            if args.command == "train":
                return _train(args, settings, console)
            if args.command == "monitor":
                return _monitor(args, target, settings, console)
            return _simulate(args, settings, console)
    except (ValueError, OSError) as exc:
        _line(Console(stderr=True), f"Error: {exc}", style="red")
        return 1
    except KeyboardInterrupt as exc:
        Console(stderr=True).print("Interrupted.")
        return _signal_status(exc)


if __name__ == "__main__":
    raise SystemExit(main())
