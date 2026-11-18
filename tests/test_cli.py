from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

import main as cli
from core.detector import CRITICAL
from core.markov_model import MarkovModel
from core.tracer import SyscallEvent
from main import main
from scripts.simulate_behavior import write_lab


def test_simulate_cli_produces_artifacts_and_alert_at_labeled_input(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "main", "simulate", "--order", "2", "--quiet", "--output-dir", str(tmp_path)],
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    truth = json.loads((tmp_path / "scenario.json").read_text(encoding="utf-8"))
    records = json.loads((tmp_path / "syscall_alerts.json").read_text(encoding="utf-8"))
    assert any(
        truth["first_unusual_index"] <= r["syscall_hids"]["window"]["end_input_index"] <= truth["last_unusual_index"]
        and CRITICAL in r["syscall_hids"]["triggers"]
        for r in records
    )
    for plot in ("markov_heatmap.png", "anomaly_timeline.png"):
        assert (tmp_path / plot).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    graph = (tmp_path / "markov_graph.dot").read_text(encoding="utf-8")
    assert graph.startswith("digraph SyscallHIDS") and 'color="#2470ae"' in graph and 'color="#bc2c44"' in graph
    assert MarkovModel.load(tmp_path / "profile.json").order == 2


def test_train_glob_calibrates_on_default_file_holdout_and_updates(tmp_path):
    lab = write_lab(tmp_path / "inputs", requests=12)
    profile = tmp_path / "profile.json"
    assert main(["train", "-i", str(lab.scenario.parent / "benign_*.jsonl"), "--model", str(profile)]) == 0
    baseline = MarkovModel.load(profile)
    count = len(lab.training[0].read_text(encoding="utf-8").splitlines())
    assert baseline.transition_count == 3 * (count - 1)
    assert baseline.calibration.samples == count - 20 + 1
    assert (
        main(
            [
                "train",
                "--update",
                str(profile),
                "-i",
                str(lab.training[-1]),
                "--validation",
                str(lab.validation[0]),
                "--model",
                str(profile),
            ]
        )
        == 0
    )
    updated = MarkovModel.load(profile)
    assert updated.transition_count == 4 * (count - 1)
    assert updated.calibration.samples == baseline.calibration.samples


def test_single_file_holdout_and_failed_retraining_preserves_profile(tmp_path):
    trace = tmp_path / "trace.jsonl"
    profile = tmp_path / "profile.json"
    calls = ["read", "close"] * 16 + ["write", "poll"] * 4
    trace.write_text("".join(SyscallEvent(call).json_line() for call in calls), encoding="utf-8")
    opts = ["train", "-i", str(trace), "--model", str(profile)]
    assert main([*opts, "--window", "5"]) == 0
    model = MarkovModel.load(profile)
    assert model.transition_count == 31
    assert model.calibration.samples == 4
    assert model.transition("close", "read").observed_count == 15
    assert "write" not in model.vocabulary
    before = profile.read_bytes()
    assert main([*opts, "--window", "9"]) == 1  # held-out stream has only 8 events
    assert profile.read_bytes() == before


def test_monitor_config_precedence_alert_stream_and_errors(tmp_path, capsys):
    lab = write_lab(tmp_path, requests=40)
    config = tmp_path / "hids.yaml"
    config.write_text("order: 3\nalpha: 0.5\nwindow: 8\n", encoding="utf-8")
    profile = tmp_path / "profile.json"
    args = [
        "--config",
        str(config),
        "train",
        "--order",
        "2",
        "-i",
        str(lab.training[0]),
        "--validation",
        str(lab.validation[0]),
        "--model",
        str(profile),
    ]
    assert main(args) == 0
    model = MarkovModel.load(profile)
    assert model.order == 2 and model.alpha == 0.5 and model.calibration.window == 8
    monitor = ["monitor", "--model", str(profile), "--input", str(lab.scenario), "--quiet"]
    log = tmp_path / "alerts.jsonl"
    assert main([*monitor, "--alerts", str(log)]) == 0
    assert all(json.loads(line)["event"]["kind"] == "alert" for line in log.read_text(encoding="utf-8").splitlines())
    capture = lab.scenario.read_bytes()
    assert main([*monitor, "--alerts", str(lab.scenario)]) == 1
    assert "overlapping" in capsys.readouterr().err
    assert lab.scenario.read_bytes() == capture
    assert main([*monitor, "--window", "10", "--alerts", str(log)]) == 1
    assert main([*monitor, "--window", "10", "--threshold", "10", "--alerts", str(log)]) == 0
    config.write_text("window: nope\n", encoding="utf-8")
    assert main(["--config", str(config), *monitor, "--alerts", str(log)]) == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["trace", "-o", "ignored.jsonl"],
        ["trace", "-o", "ignored.jsonl", "--pid", "3", "--", "unused"],
        ["monitor", "--model", "profile.json", "--input", "capture.jsonl", "--pid", "3"],
        ["monitor", "--model", "profile.json", "--input", "capture.jsonl", "--plot-dir", "unused"],
    ],
)
def test_invalid_target_or_option_combinations_are_usage_errors(argv):
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 2


def test_standalone_generator(tmp_path: Path):
    target = tmp_path / "generated"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.simulate_behavior",
            "--seed",
            "1",
            "--requests",
            "4",
            "--output-dir",
            str(target),
        ],
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert len(list(target.glob("*.jsonl"))) == 7
    assert (target / "scenario.json").is_file()
    assert not (target / "profile.json").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX termination signal")
def test_graceful_termination_keeps_partial_capture(tmp_path, monkeypatch):
    target = tmp_path / "partial.jsonl"
    original_handler = signal.getsignal(signal.SIGTERM)

    def capture(*_):
        yield SyscallEvent("read")
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler) and handler is not original_handler
        handler(signal.SIGTERM, None)

    monkeypatch.setattr(cli, "_live", capture)
    assert main(["trace", "-o", str(target), "--pid", "5"]) == 143
    assert json.loads(target.read_text(encoding="utf-8")) == {"token": "read"}
    assert signal.getsignal(signal.SIGTERM) is original_handler
